# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import copy
import colorsys
import hashlib
import math
import random
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vector_norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vec, vec))


def _difference_score(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    diff = (left[0] - right[0], left[1] - right[1], left[2] - right[2])
    return _vector_norm(diff) / math.sqrt(3.0)


def _four_bin_code(values: np.ndarray | list[float] | list[int]) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    size = int(arr.size)
    if size <= 0:
        return "0000"
    tokens: list[str] = []
    start = 0
    for bucket in range(4):
        end = ((bucket + 1) * size) // 4
        if bucket == 3:
            end = size
        if end <= start:
            tokens.append("0")
            continue
        part = arr[start:end]
        mean = float(part.sum()) / max(1, part.size)
        tokens.append(str(int(max(0, min(3, math.floor(_clamp(mean, 0.0, 0.9999) * 4.0))))))
        start = end
    return "".join((tokens + ["0", "0", "0", "0"])[:4])


def _clone_sa_item(item: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(item)
    if "coords" in cloned:
        cloned["coords"] = dict(item.get("coords", {}) or {})
    if "attributes" in cloned:
        cloned["attributes"] = dict(item.get("attributes", {}) or {})
    return cloned


class VisionSensorV1:
    def __init__(
        self,
        *,
        patch_budget: int,
        focus_patch_budget: int,
        reconstruction_patch_budget: int | None = None,
        raw_state_budget: int | None = None,
        edge_candidate_gain: float = 1.0,
        edge_priority_gain: float = 1.0,
        attention_boost_enabled: bool = False,
        attention_boost_decay: float = 0.72,
        attention_boost_max_extra_raw_budget: int = 0,
        attention_boost_max_extra_focus_budget: int = 0,
        attention_boost_min_radius_scale: float = 0.35,
        attention_boost_edge_gain: float = 1.0,
        attention_boost_gaze_sigma_scale: float = 0.7,
        dynamic_track_window: int = 6,
        dynamic_candidate_limit_background: int = 12,
        dynamic_candidate_limit_focus: int = 28,
        dynamic_track_limit: int = 40,
        dynamic_summary_limit: int = 4,
        dynamic_match_threshold: float = 0.46,
        dynamic_track_forget_ticks: int = 3,
        export_preview_image: bool = True,
    ) -> None:
        self.patch_budget = max(4, int(patch_budget))
        self.focus_patch_budget = max(1, int(focus_patch_budget))
        default_raw_state_budget = max(self.patch_budget * 8, self.focus_patch_budget * 8, 64)
        self.raw_state_budget = max(self.patch_budget, int(raw_state_budget or reconstruction_patch_budget or default_raw_state_budget))
        self.reconstruction_patch_budget = max(self.raw_state_budget, int(reconstruction_patch_budget or self.raw_state_budget))
        self.edge_candidate_gain = max(0.0, float(edge_candidate_gain))
        self.edge_priority_gain = max(0.0, float(edge_priority_gain))
        self.attention_boost_enabled = bool(attention_boost_enabled)
        self.attention_boost_decay = _clamp(float(attention_boost_decay), 0.0, 1.0)
        self.attention_boost_max_extra_raw_budget = max(0, int(attention_boost_max_extra_raw_budget))
        self.attention_boost_max_extra_focus_budget = max(0, int(attention_boost_max_extra_focus_budget))
        self.attention_boost_min_radius_scale = _clamp(float(attention_boost_min_radius_scale), 0.05, 1.0)
        self.attention_boost_edge_gain = max(0.0, float(attention_boost_edge_gain))
        self.attention_boost_gaze_sigma_scale = _clamp(float(attention_boost_gaze_sigma_scale), 0.05, 2.0)
        self.dynamic_track_window = max(2, int(dynamic_track_window))
        self.dynamic_candidate_limit_background = max(2, int(dynamic_candidate_limit_background))
        self.dynamic_candidate_limit_focus = max(self.dynamic_candidate_limit_background, int(dynamic_candidate_limit_focus))
        self.dynamic_track_limit = max(4, int(dynamic_track_limit))
        self.dynamic_summary_limit = max(1, int(dynamic_summary_limit))
        self.dynamic_match_threshold = _clamp(float(dynamic_match_threshold), 0.05, 0.95)
        self.dynamic_track_forget_ticks = max(1, int(dynamic_track_forget_ticks))
        self.export_preview_image = bool(export_preview_image)
        self.gaze_center = (0.5, 0.5)
        self._sensor_tick = 0
        self._stream_frame_index = -1
        self._prev_raw_samples: dict[tuple[int, int], dict[str, float]] = {}
        self._recent_selected_counts: dict[tuple[int, int], int] = {}
        self._fixation_buffer: dict[tuple[int, int], dict[str, Any]] = {}
        self._dynamic_shape_tracks: dict[str, dict[str, Any]] = {}
        self._recent_shape_candidate_ring: list[list[dict[str, Any]]] = []
        self._global_motion_history: list[dict[str, float]] = []
        self._dynamic_track_serial = 0
        self._last_preview_size = (0, 0)
        self._visual_frame_cache: dict[str, dict[str, Any]] = {}
        self._visual_frame_cache_limit = 8
        self._visual_patch_static_cache: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        self._visual_patch_static_cache_limit = 4
        self._pixel_access: Any | None = None
        self._gray_pixel_access: Any | None = None
        self._gray_image: Image.Image | None = None
        self._gray_array: np.ndarray | None = None
        self._rgb_array: np.ndarray | None = None
        self._prev_frame_gray_u8: np.ndarray | None = None
        self._prev_frame_rgb_u8: np.ndarray | None = None
        self._local_patch_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._attention_boost: dict[str, Any] = {
            "active": False,
            "strength": 0.0,
            "ticks_left": 0,
            "target_gaze": {"x": 0.5, "y": 0.5},
            "source_action": "",
            "raw_budget_bonus": 0,
            "focus_budget_bonus": 0,
            "radius_scale": 1.0,
            "edge_gain": 1.0,
            "gaze_sigma_scale": 1.0,
        }
        self._attention_mode = "background"

    def set_attention_mode(self, mode: str) -> None:
        clean = str(mode or "").strip().lower()
        if clean not in {"background", "suppressed", "visual_focus"}:
            clean = "background"
        self._attention_mode = clean

    def apply_attention_boost(
        self,
        *,
        source_action: str,
        firmness_norm: float,
        target_gaze: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        if not self.attention_boost_enabled:
            return self.attention_boost_snapshot()
        strength = _clamp(float(firmness_norm), 0.0, 1.5)
        if strength <= 0.0:
            return self.attention_boost_snapshot()
        target = target_gaze or self.gaze_center
        strength_norm = _clamp(strength / 1.0, 0.0, 1.0)
        raw_bonus = int(round(self.attention_boost_max_extra_raw_budget * strength_norm))
        focus_bonus = int(round(self.attention_boost_max_extra_focus_budget * strength_norm))
        radius_scale = max(self.attention_boost_min_radius_scale, 1.0 - (1.0 - self.attention_boost_min_radius_scale) * strength_norm)
        gaze_sigma_scale = max(0.05, 1.0 - (1.0 - self.attention_boost_gaze_sigma_scale) * strength_norm)
        edge_gain = 1.0 + max(0.0, self.attention_boost_edge_gain - 1.0) * strength_norm
        ticks_left = max(1, int(round(1 + strength_norm * 2.0)))
        self._attention_boost = {
            "active": True,
            "strength": _round4(strength_norm),
            "ticks_left": ticks_left,
            "target_gaze": {"x": _round4(_clamp(target[0], 0.0, 1.0)), "y": _round4(_clamp(target[1], 0.0, 1.0))},
            "source_action": str(source_action or ""),
            "raw_budget_bonus": int(raw_bonus),
            "focus_budget_bonus": int(focus_bonus),
            "radius_scale": _round4(radius_scale),
            "edge_gain": _round4(edge_gain),
            "gaze_sigma_scale": _round4(gaze_sigma_scale),
        }
        return self.attention_boost_snapshot()

    def attention_boost_snapshot(self) -> dict[str, Any]:
        return {
            "active": bool(self._attention_boost.get("active", False)),
            "strength": _round4(float(self._attention_boost.get("strength", 0.0) or 0.0)),
            "ticks_left": int(self._attention_boost.get("ticks_left", 0) or 0),
            "target_gaze": dict(self._attention_boost.get("target_gaze", {}) or {"x": 0.5, "y": 0.5}),
            "source_action": str(self._attention_boost.get("source_action", "") or ""),
            "raw_budget_bonus": int(self._attention_boost.get("raw_budget_bonus", 0) or 0),
            "focus_budget_bonus": int(self._attention_boost.get("focus_budget_bonus", 0) or 0),
            "radius_scale": _round4(float(self._attention_boost.get("radius_scale", 1.0) or 1.0)),
            "edge_gain": _round4(float(self._attention_boost.get("edge_gain", 1.0) or 1.0)),
            "gaze_sigma_scale": _round4(float(self._attention_boost.get("gaze_sigma_scale", 1.0) or 1.0)),
            "attention_mode": str(self._attention_mode or "background"),
        }

    def _effective_sampling_profile(self) -> dict[str, Any]:
        boost = self.attention_boost_snapshot()
        background_raw_budget = max(16, min(int(self.raw_state_budget), 64))
        background_patch_budget = max(4, min(self.patch_budget, 8))
        background_focus_budget = max(2, min(self.focus_patch_budget, background_patch_budget))
        if not boost.get("active"):
            if str(self._attention_mode or "background") == "suppressed":
                suppressed_raw_budget = max(16, min(int(self.raw_state_budget), 32))
                suppressed_patch_budget = max(4, min(self.patch_budget, 6))
                suppressed_focus_budget = max(2, min(self.focus_patch_budget, 3, suppressed_patch_budget))
                return {
                    "raw_budget": int(suppressed_raw_budget),
                    "focus_priority_budget": int(min(suppressed_focus_budget, suppressed_patch_budget, suppressed_raw_budget)),
                    "memory_write_budget": int(min(suppressed_patch_budget, suppressed_raw_budget)),
                    "radius_scale": 1.0,
                    "edge_gain": max(0.0, float(self.edge_priority_gain)),
                    "candidate_edge_gain": max(0.0, float(self.edge_candidate_gain)),
                    "gaze_sigma_scale": 1.0,
                    "attention_mode": "suppressed",
                    "boost": boost,
                }
            return {
                "raw_budget": int(background_raw_budget),
                "focus_priority_budget": int(min(background_focus_budget, background_patch_budget, background_raw_budget)),
                "memory_write_budget": int(min(background_patch_budget, background_raw_budget)),
                "radius_scale": 1.0,
                "edge_gain": max(0.0, float(self.edge_priority_gain)),
                "candidate_edge_gain": max(0.0, float(self.edge_candidate_gain)),
                "gaze_sigma_scale": 1.0,
                "attention_mode": "background",
                "boost": boost,
            }
        boost_requested = background_raw_budget + int(boost.get("raw_budget_bonus", 0) or 0)
        boost_cap = min(
            256,
            max(
                background_raw_budget * 4,
                self.patch_budget * 16,
                self.focus_patch_budget * 24,
            ),
        )
        raw_budget = max(4, min(int(boost_requested), int(boost_cap)))
        focus_priority_budget = min(
            self.patch_budget,
            raw_budget,
            int(self.focus_patch_budget + int(boost.get("focus_budget_bonus", 0) or 0)),
        )
        return {
            "raw_budget": raw_budget,
            "focus_priority_budget": int(max(1, focus_priority_budget)),
            "memory_write_budget": int(min(self.patch_budget, raw_budget)),
            "radius_scale": _clamp(float(boost.get("radius_scale", 1.0) or 1.0), 0.05, 1.0),
            "edge_gain": max(0.0, float(self.edge_priority_gain) * float(boost.get("edge_gain", 1.0) or 1.0)),
            "candidate_edge_gain": max(0.0, float(self.edge_candidate_gain) * float(boost.get("edge_gain", 1.0) or 1.0)),
            "gaze_sigma_scale": _clamp(float(boost.get("gaze_sigma_scale", 1.0) or 1.0), 0.05, 2.0),
            "attention_mode": "visual_focus",
            "boost": boost,
        }

    def _decay_attention_boost(self) -> None:
        if not bool(self._attention_boost.get("active", False)):
            return
        ticks_left = max(0, int(self._attention_boost.get("ticks_left", 0) or 0) - 1)
        strength = max(0.0, float(self._attention_boost.get("strength", 0.0) or 0.0) * self.attention_boost_decay)
        self._attention_boost["ticks_left"] = ticks_left
        self._attention_boost["strength"] = _round4(strength)
        if ticks_left <= 0 or strength <= 0.01:
            self._attention_boost = {
                "active": False,
                "strength": 0.0,
                "ticks_left": 0,
                "target_gaze": dict(self._attention_boost.get("target_gaze", {}) or {"x": self.gaze_center[0], "y": self.gaze_center[1]}),
                "source_action": str(self._attention_boost.get("source_action", "") or ""),
                "raw_budget_bonus": 0,
                "focus_budget_bonus": 0,
                "radius_scale": 1.0,
                "edge_gain": 1.0,
                "gaze_sigma_scale": 1.0,
            }

    def ingest_image_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "image_input") -> dict[str, Any]:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        width, height = image.size
        self._last_preview_size = (width, height)
        self._sensor_tick += 1
        self._stream_frame_index += 1
        self._pixel_access = image.load()
        self._gray_image = image.convert("L")
        self._gray_pixel_access = self._gray_image.load() if self._gray_image is not None else None
        self._gray_array = np.asarray(self._gray_image, dtype=np.float32) if self._gray_image is not None else None
        self._rgb_array = np.asarray(image, dtype=np.uint8)
        self._local_patch_cache = {}
        frame_cache_key = self._visual_frame_cache_key(raw_bytes)
        cached_patch_map = self._get_cached_patch_static_map(frame_cache_key) if frame_cache_key else None
        if cached_patch_map is not None:
            self._local_patch_cache = {
                (int(key[0]), int(key[1])): dict(value)
                for key, value in cached_patch_map.items()
            }
        cached_visual = self._get_cached_visual_frame(frame_cache_key, source_type=source_type) if frame_cache_key else None

        if cached_visual is not None:
            preview_payload = dict(cached_visual.get("preview_payload", {}) or {})
            preview_payload["width"] = width
            preview_payload["height"] = height
            if not self.export_preview_image:
                preview_payload["data_url"] = ""
            contour_bundle = dict(cached_visual.get("contour_bundle", {}) or {})
        else:
            preview_payload = {"data_url": "", "width": width, "height": height}
            if self.export_preview_image:
                preview_payload["data_url"] = self._encode_preview_data_url(image)
            contour_bundle = self._build_contour_bundle(image=image, source_type=source_type)
            if frame_cache_key:
                self._store_cached_visual_frame(
                    frame_cache_key,
                    preview_payload=preview_payload,
                    contour_bundle=contour_bundle,
                )
        motion_bundle = self._build_motion_contour_bundle(image=image)
        contour_bundle = self._merge_contour_with_motion_bundle(
            contour_bundle=contour_bundle,
            motion_bundle=motion_bundle,
            source_type=source_type,
        )

        sampling_profile = self._effective_sampling_profile()
        raw_sample_budget = int(sampling_profile["raw_budget"])
        focus_priority_budget = int(sampling_profile["focus_priority_budget"])
        memory_write_budget = int(sampling_profile["memory_write_budget"])

        raw_samples, selected_samples, focus_samples, motion_values = self._sample_original_resolution(
            image=image,
            sample_count=raw_sample_budget,
            focus_priority_budget=focus_priority_budget,
            memory_write_budget=memory_write_budget,
            source_type=source_type,
            sampling_profile=sampling_profile,
        )
        global_structure_samples = self._build_global_structure_samples(
            image=image,
            source_type=source_type,
            contour_bundle=contour_bundle,
        )
        self._update_fixation_buffer(raw_samples, tick_index=tick_index)
        shape_candidates = self._build_shape_candidates(
            image=image,
            raw_samples=raw_samples,
            global_structure_samples=global_structure_samples,
            contour_bundle=contour_bundle,
            source_type=source_type,
        )
        global_motion = self._estimate_global_motion(shape_candidates)
        dynamic_tracks, dynamic_motion_samples, dynamic_summary = self._update_dynamic_tracks(
            tick_index=tick_index,
            shape_candidates=shape_candidates,
            global_motion=global_motion,
            source_type=source_type,
        )
        self._push_recent_shape_candidates(shape_candidates)

        reconstruction_cells = [
            {
                "row": int(sample["coords"].get("pixel_y", 0) or 0),
                "col": int(sample["coords"].get("pixel_x", 0) or 0),
                "pixel_x": int(sample["coords"].get("pixel_x", 0) or 0),
                "pixel_y": int(sample["coords"].get("pixel_y", 0) or 0),
                "screen_x": _round4(float(sample["coords"].get("screen_x", 0.0) or 0.0)),
                "screen_y": _round4(float(sample["coords"].get("screen_y", 0.0) or 0.0)),
                "screen_w": _round4(float(sample["coords"].get("screen_w", 0.0) or 0.0)),
                "screen_h": _round4(float(sample["coords"].get("screen_h", 0.0) or 0.0)),
                "cx": _round4(float(sample["coords"].get("cx", 0.0) or 0.0)),
                "cy": _round4(float(sample["coords"].get("cy", 0.0) or 0.0)),
                "avg_r": _round4(float((sample["attributes"] or {}).get("avg_r", 0.0) or 0.0)),
                "avg_g": _round4(float((sample["attributes"] or {}).get("avg_g", 0.0) or 0.0)),
                "avg_b": _round4(float((sample["attributes"] or {}).get("avg_b", 0.0) or 0.0)),
                "brightness": _round4(float((sample["attributes"] or {}).get("brightness", 0.0) or 0.0)),
                "motion": _round4(float((sample["attributes"] or {}).get("motion", 0.0) or 0.0)),
                "energy": _round4(float(sample.get("energy", 0.0) or 0.0)),
                "sample_reason": str((sample["attributes"] or {}).get("sample_reason", "raw") or "raw"),
            }
            for sample in raw_samples
        ]
        fixation_cells = self._export_fixation_cells(width=width, height=height)

        packet = {
            "schema_id": "vision_sensor_packet/v1",
            "schema_version": "2.1",
            "sensor_name": "vision_sensor_v1",
            "tick_index": int(tick_index),
            "sensor_tick": self._sensor_tick,
            "source_type": source_type,
            "image_size": {"width": width, "height": height},
            "preview_image": {
                **preview_payload,
            },
            "contour_reconstruction": contour_bundle,
            "gaze_center": {"x": _round4(self.gaze_center[0]), "y": _round4(self.gaze_center[1])},
            "gaze_point": {
                "pixel_x": int(round(self.gaze_center[0] * max(0, width - 1))),
                "pixel_y": int(round(self.gaze_center[1] * max(0, height - 1))),
                "x": _round4(self.gaze_center[0]),
                "y": _round4(self.gaze_center[1]),
            },
            "grid": {"cols": width, "rows": height},
            "budget_used": len(selected_samples),
            "cognitive_patch_budget": self.patch_budget,
            "total_patch_count": len(raw_samples),
            "raw_state_budget": raw_sample_budget,
            "raw_sample_budget": raw_sample_budget,
            "focus_priority_budget": focus_priority_budget,
            "memory_write_budget": memory_write_budget,
            "focus_memory_write_budget": min(len(focus_samples), focus_priority_budget),
            "reconstruction_patch_budget": self.reconstruction_patch_budget,
            "attention_boost": sampling_profile["boost"],
            "reconstruction_grid": {
                "cols": width,
                "rows": height,
                "cell_count": len(reconstruction_cells),
                "cells": reconstruction_cells,
                "mode": "original_resolution_sparse_overlay",
                "dense_cells_externalized": False,
            },
            "patches": selected_samples,
            "memory_write_samples": selected_samples,
            "raw_samples": raw_samples,
            "focus_priority_samples": focus_samples,
            "global_structure_samples": global_structure_samples,
            "shape_candidates": shape_candidates,
            "dynamic_tracks": dynamic_tracks,
            "dynamic_motion_samples": dynamic_motion_samples,
            "dynamic_track_summary": dynamic_summary,
            "global_structure_summary": {
                "count": len(global_structure_samples),
                "preview": [
                    str(((item.get("attributes", {}) or {}).get("global_feature_code", "") or item.get("sa_label", "") or ""))
                    for item in global_structure_samples[:6]
                ],
            },
            "fixation_buffer": {
                "cell_count": len(fixation_cells),
                "cells": fixation_cells,
                "tracked_sample_count": len(self._fixation_buffer),
            },
            "stream_state": {
                "frame_index": self._stream_frame_index,
                "prev_frame_available": bool(self._sensor_tick > 1),
                "selected_focus_budget": focus_priority_budget,
                "motion_mean": _round4(sum(motion_values) / max(1, len(motion_values))),
                "global_motion_dx": _round4(float(global_motion.get("dx", 0.0) or 0.0)),
                "global_motion_dy": _round4(float(global_motion.get("dy", 0.0) or 0.0)),
                "global_motion_speed": _round4(float(global_motion.get("speed", 0.0) or 0.0)),
                "tracked_patch_count": len(self._prev_raw_samples),
                "dense_grid_cell_count": len(reconstruction_cells),
                "fixation_buffer_count": len(self._fixation_buffer),
                "raw_state_budget": raw_sample_budget,
                "memory_write_budget": memory_write_budget,
                "effective_focus_priority_budget": focus_priority_budget,
                "global_structure_count": len(global_structure_samples),
                "shape_candidate_count": len(shape_candidates),
                "dynamic_track_count": int(dynamic_summary.get("track_count", 0) or 0),
                "dynamic_object_count": int(dynamic_summary.get("object_count", 0) or 0),
                "dynamic_salience_mean": _round4(float(dynamic_summary.get("dynamic_salience_mean", 0.0) or 0.0)),
                "attention_boost_active": bool((sampling_profile["boost"] or {}).get("active", False)),
                "edge_priority_gain": _round4(float(sampling_profile.get("edge_gain", self.edge_priority_gain) or self.edge_priority_gain)),
                "edge_candidate_gain": _round4(float(sampling_profile.get("candidate_edge_gain", self.edge_candidate_gain) or self.edge_candidate_gain)),
            },
        }
        if self._gray_image is not None:
            self._prev_frame_gray_u8 = np.asarray(self._gray_image, dtype=np.uint8).copy()
        else:
            self._prev_frame_gray_u8 = None
        if self._rgb_array is not None:
            self._prev_frame_rgb_u8 = np.asarray(self._rgb_array, dtype=np.uint8).copy()
        else:
            self._prev_frame_rgb_u8 = None
        self._decay_attention_boost()
        self._pixel_access = None
        self._gray_pixel_access = None
        self._gray_image = None
        self._gray_array = None
        self._rgb_array = None
        if frame_cache_key:
            export_patch_map: dict[tuple[int, int], dict[str, Any]] = {}
            for key, value in self._local_patch_cache.items():
                if not isinstance(key, tuple) or len(key) != 2 or not isinstance(value, dict):
                    continue
                cloned = {k: v for k, v in value.items() if k != "bright_at"}
                export_patch_map[(int(key[0]), int(key[1]))] = cloned
            self._store_cached_patch_static_map(frame_cache_key, export_patch_map)
        self._local_patch_cache = {}
        return packet

    def _frame_change_score_for_coords(self, coords: dict[str, Any]) -> float:
        current_gray_arr = self._gray_array
        prev_gray = self._prev_frame_gray_u8
        current_rgb = self._rgb_array
        prev_rgb = self._prev_frame_rgb_u8
        if current_gray_arr is None or prev_gray is None or current_rgb is None or prev_rgb is None:
            return 0.0
        if current_rgb.shape != prev_rgb.shape or current_gray_arr.shape != prev_gray.shape:
            return 0.0
        height, width = current_rgb.shape[:2]
        x0 = int(math.floor(float(coords.get("screen_x", 0.0) or 0.0) * width))
        y0 = int(math.floor(float(coords.get("screen_y", 0.0) or 0.0) * height))
        w = max(1, int(math.ceil(float(coords.get("screen_w", 0.0) or 0.0) * width)))
        h = max(1, int(math.ceil(float(coords.get("screen_h", 0.0) or 0.0) * height)))
        pad_x = max(1, int(round(w * 0.12)))
        pad_y = max(1, int(round(h * 0.12)))
        left = max(0, x0 - pad_x)
        top = max(0, y0 - pad_y)
        right = min(width, x0 + w + pad_x)
        bottom = min(height, y0 + h + pad_y)
        if right <= left or bottom <= top:
            return 0.0
        gray_region = current_gray_arr[top:bottom, left:right].astype(np.float32)
        prev_gray_region = prev_gray[top:bottom, left:right].astype(np.float32)
        current_rgb_region = current_rgb[top:bottom, left:right].astype(np.float32)
        prev_rgb_region = prev_rgb[top:bottom, left:right].astype(np.float32)
        if gray_region.size <= 0 or current_rgb_region.size <= 0:
            return 0.0
        gray_diff = np.abs(gray_region - prev_gray_region) / 255.0
        rgb_diff = np.mean(np.abs(current_rgb_region - prev_rgb_region), axis=2) / 255.0
        mean_change = float(np.mean(rgb_diff)) if rgb_diff.size else 0.0
        salient_change = float(np.percentile(rgb_diff, 85)) if rgb_diff.size else 0.0
        luma_change = float(np.mean(gray_diff)) if gray_diff.size else 0.0
        score = mean_change * 0.44 + salient_change * 0.36 + luma_change * 0.20
        return _clamp(score * 2.2, 0.0, 1.0)

    def _visual_frame_cache_key(self, raw_bytes: bytes) -> str:
        payload = bytes(raw_bytes or b"")
        if not payload:
            return ""
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _update_contour_bundle_gaze_coords(self, contour_bundle: dict[str, Any], *, source_type: str) -> dict[str, Any]:
        cloned = copy.deepcopy(contour_bundle or {})
        if not cloned:
            return cloned
        cloned["source_type"] = source_type
        for component in cloned.get("components", []) or []:
            if not isinstance(component, dict):
                continue
            coords = component.get("coords", {}) or {}
            if not isinstance(coords, dict):
                continue
            cx = _clamp(float(coords.get("cx", 0.5) or 0.5), 0.0, 1.0)
            cy = _clamp(float(coords.get("cy", 0.5) or 0.5), 0.0, 1.0)
            coords["dx_from_gaze"] = _round4(cx - self.gaze_center[0])
            coords["dy_from_gaze"] = _round4(cy - self.gaze_center[1])
            coords["dr_from_gaze"] = _round4(math.sqrt((cx - self.gaze_center[0]) ** 2 + (cy - self.gaze_center[1]) ** 2))
            component["coords"] = coords
        return cloned

    def _get_cached_visual_frame(self, cache_key: str, *, source_type: str) -> dict[str, Any] | None:
        key = str(cache_key or "")
        if not key:
            return None
        cached = self._visual_frame_cache.pop(key, None)
        if cached is None:
            return None
        self._visual_frame_cache[key] = cached
        preview_payload = dict(cached.get("preview_payload", {}) or {})
        contour_bundle = self._update_contour_bundle_gaze_coords(
            dict(cached.get("contour_bundle", {}) or {}),
            source_type=source_type,
        )
        return {
            "preview_payload": preview_payload,
            "contour_bundle": contour_bundle,
        }

    def _store_cached_visual_frame(self, cache_key: str, *, preview_payload: dict[str, Any], contour_bundle: dict[str, Any]) -> None:
        key = str(cache_key or "")
        if not key:
            return
        self._visual_frame_cache[key] = {
            "preview_payload": dict(preview_payload or {}),
            "contour_bundle": copy.deepcopy(contour_bundle or {}),
        }
        while len(self._visual_frame_cache) > self._visual_frame_cache_limit:
            oldest_key = next(iter(self._visual_frame_cache))
            if oldest_key == key and len(self._visual_frame_cache) == 1:
                break
            self._visual_frame_cache.pop(oldest_key, None)

    def _get_cached_patch_static_map(self, cache_key: str) -> dict[tuple[int, int], dict[str, Any]] | None:
        key = str(cache_key or "")
        if not key:
            return None
        cached = self._visual_patch_static_cache.pop(key, None)
        if cached is None:
            return None
        self._visual_patch_static_cache[key] = cached
        return cached

    def _store_cached_patch_static_map(self, cache_key: str, payload: dict[tuple[int, int], dict[str, Any]]) -> None:
        key = str(cache_key or "")
        if not key:
            return
        self._visual_patch_static_cache[key] = payload
        while len(self._visual_patch_static_cache) > self._visual_patch_static_cache_limit:
            oldest_key = next(iter(self._visual_patch_static_cache))
            if oldest_key == key and len(self._visual_patch_static_cache) == 1:
                break
            self._visual_patch_static_cache.pop(oldest_key, None)

    def move_gaze(self, x: float, y: float) -> None:
        self.gaze_center = (_clamp(float(x), 0.0, 1.0), _clamp(float(y), 0.0, 1.0))

    def export_payload(self) -> dict[str, Any]:
        fixation_payload: list[dict[str, Any]] = []
        for (pixel_x, pixel_y), row in self._fixation_buffer.items():
            fixation_payload.append(
                {
                    "pixel_x": int(pixel_x),
                    "pixel_y": int(pixel_y),
                    "energy": _round4(float(row.get("energy", 0.0) or 0.0)),
                    "brightness": _round4(float(row.get("brightness", 0.0) or 0.0)),
                    "avg_r": _round4(float(row.get("avg_r", 0.0) or 0.0)),
                    "avg_g": _round4(float(row.get("avg_g", 0.0) or 0.0)),
                    "avg_b": _round4(float(row.get("avg_b", 0.0) or 0.0)),
                    "last_seen_tick": int(row.get("last_seen_tick", -1) or -1),
                    "sample_hits": int(row.get("sample_hits", 0) or 0),
                    "source_tag": str(row.get("source_tag", "") or ""),
                    "sample_reason": str(row.get("sample_reason", "") or ""),
                }
            )
        return {
            "sensor_tick": self._sensor_tick,
            "gaze_center": {"x": self.gaze_center[0], "y": self.gaze_center[1]},
            "prev_raw_samples": {
                f"{pixel_x}:{pixel_y}": {
                    "brightness": _round4(float(row.get("brightness", 0.0) or 0.0)),
                    "avg_r": _round4(float(row.get("avg_r", 0.0) or 0.0)),
                    "avg_g": _round4(float(row.get("avg_g", 0.0) or 0.0)),
                    "avg_b": _round4(float(row.get("avg_b", 0.0) or 0.0)),
                }
                for (pixel_x, pixel_y), row in self._prev_raw_samples.items()
            },
            "stream_frame_index": self._stream_frame_index,
            "recent_selected_counts": {f"{pixel_x}:{pixel_y}": int(value) for (pixel_x, pixel_y), value in self._recent_selected_counts.items()},
            "fixation_buffer": fixation_payload,
            "dynamic_shape_tracks": [_clone_sa_item(row) for row in self._dynamic_shape_tracks.values()],
            "recent_shape_candidate_ring": [[_clone_sa_item(item) for item in frame] for frame in self._recent_shape_candidate_ring],
            "global_motion_history": [dict(row) for row in self._global_motion_history],
            "dynamic_track_serial": int(self._dynamic_track_serial),
            "raw_state_budget": self.raw_state_budget,
            "reconstruction_patch_budget": self.reconstruction_patch_budget,
            "last_preview_size": {"width": int(self._last_preview_size[0]), "height": int(self._last_preview_size[1])},
            "edge_candidate_gain": _round4(self.edge_candidate_gain),
            "edge_priority_gain": _round4(self.edge_priority_gain),
            "attention_boost_enabled": bool(self.attention_boost_enabled),
            "attention_boost_decay": _round4(self.attention_boost_decay),
            "attention_boost_max_extra_raw_budget": int(self.attention_boost_max_extra_raw_budget),
            "attention_boost_max_extra_focus_budget": int(self.attention_boost_max_extra_focus_budget),
            "attention_boost_min_radius_scale": _round4(self.attention_boost_min_radius_scale),
            "attention_boost_edge_gain": _round4(self.attention_boost_edge_gain),
            "attention_boost_gaze_sigma_scale": _round4(self.attention_boost_gaze_sigma_scale),
            "dynamic_track_window": int(self.dynamic_track_window),
            "dynamic_candidate_limit_background": int(self.dynamic_candidate_limit_background),
            "dynamic_candidate_limit_focus": int(self.dynamic_candidate_limit_focus),
            "dynamic_track_limit": int(self.dynamic_track_limit),
            "dynamic_summary_limit": int(self.dynamic_summary_limit),
            "dynamic_match_threshold": _round4(self.dynamic_match_threshold),
            "dynamic_track_forget_ticks": int(self.dynamic_track_forget_ticks),
            "export_preview_image": bool(self.export_preview_image),
            "attention_boost": self.attention_boost_snapshot(),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._sensor_tick = int(payload.get("sensor_tick", 0) or 0)
        gaze = payload.get("gaze_center", {}) or {}
        self.move_gaze(float(gaze.get("x", 0.5) or 0.5), float(gaze.get("y", 0.5) or 0.5))
        raw_prev = payload.get("prev_raw_samples", {}) or {}
        self._prev_raw_samples = {}
        for key, value in raw_prev.items():
            if not isinstance(value, dict):
                continue
            clean = str(key or "")
            if ":" not in clean:
                continue
            left, right = clean.split(":", 1)
            try:
                pixel_x = int(left)
                pixel_y = int(right)
            except Exception:
                continue
            self._prev_raw_samples[(pixel_x, pixel_y)] = {
                "brightness": float((value or {}).get("brightness", 0.0) or 0.0),
                "avg_r": float((value or {}).get("avg_r", 0.0) or 0.0),
                "avg_g": float((value or {}).get("avg_g", 0.0) or 0.0),
                "avg_b": float((value or {}).get("avg_b", 0.0) or 0.0),
            }
        self._stream_frame_index = int(payload.get("stream_frame_index", -1) or -1)
        recent_counts = payload.get("recent_selected_counts", {}) or {}
        self._recent_selected_counts = {}
        for key, value in recent_counts.items():
            clean = str(key or "")
            if ":" not in clean:
                continue
            left, right = clean.split(":", 1)
            try:
                pixel_x = int(left)
                pixel_y = int(right)
            except Exception:
                continue
            self._recent_selected_counts[(pixel_x, pixel_y)] = max(0, int(value or 0))
        fixation_payload = payload.get("fixation_buffer", []) or []
        self._fixation_buffer = {}
        for row in fixation_payload:
            if not isinstance(row, dict):
                continue
            pixel_x = int(row.get("pixel_x", -1) or -1)
            pixel_y = int(row.get("pixel_y", -1) or -1)
            if pixel_x < 0 or pixel_y < 0:
                continue
            self._fixation_buffer[(pixel_x, pixel_y)] = {
                "energy": float(row.get("energy", 0.0) or 0.0),
                "brightness": float(row.get("brightness", 0.0) or 0.0),
                "avg_r": float(row.get("avg_r", 0.0) or 0.0),
                "avg_g": float(row.get("avg_g", 0.0) or 0.0),
                "avg_b": float(row.get("avg_b", 0.0) or 0.0),
                "last_seen_tick": int(row.get("last_seen_tick", -1) or -1),
                "sample_hits": int(row.get("sample_hits", 0) or 0),
                "source_tag": str(row.get("source_tag", "") or ""),
                "sample_reason": str(row.get("sample_reason", "") or ""),
            }
        self._dynamic_shape_tracks = {}
        for row in payload.get("dynamic_shape_tracks", []) or []:
            if not isinstance(row, dict):
                continue
            track_id = str(row.get("track_id", "") or "")
            if not track_id:
                continue
            self._dynamic_shape_tracks[track_id] = _clone_sa_item(row)
        self._recent_shape_candidate_ring = [
            [_clone_sa_item(item) for item in frame if isinstance(item, dict)]
            for frame in (payload.get("recent_shape_candidate_ring", []) or [])
            if isinstance(frame, list)
        ][-self.dynamic_track_window :]
        self._global_motion_history = [
            {
                "dx": float((row or {}).get("dx", 0.0) or 0.0),
                "dy": float((row or {}).get("dy", 0.0) or 0.0),
                "speed": float((row or {}).get("speed", 0.0) or 0.0),
            }
            for row in (payload.get("global_motion_history", []) or [])
            if isinstance(row, dict)
        ][-self.dynamic_track_window :]
        self._dynamic_track_serial = max(0, int(payload.get("dynamic_track_serial", 0) or 0))
        if "raw_state_budget" in payload:
            self.raw_state_budget = max(self.patch_budget, int(payload.get("raw_state_budget", self.raw_state_budget) or self.raw_state_budget))
        if "reconstruction_patch_budget" in payload:
            self.reconstruction_patch_budget = max(self.raw_state_budget, int(payload.get("reconstruction_patch_budget", self.reconstruction_patch_budget) or self.reconstruction_patch_budget))
        preview_size = payload.get("last_preview_size", {}) or {}
        self._last_preview_size = (int(preview_size.get("width", 0) or 0), int(preview_size.get("height", 0) or 0))
        if "edge_candidate_gain" in payload:
            self.edge_candidate_gain = max(0.0, float(payload.get("edge_candidate_gain", self.edge_candidate_gain) or self.edge_candidate_gain))
        if "edge_priority_gain" in payload:
            self.edge_priority_gain = max(0.0, float(payload.get("edge_priority_gain", self.edge_priority_gain) or self.edge_priority_gain))
        if "attention_boost_enabled" in payload:
            self.attention_boost_enabled = bool(payload.get("attention_boost_enabled", self.attention_boost_enabled))
        if "attention_boost_decay" in payload:
            self.attention_boost_decay = _clamp(float(payload.get("attention_boost_decay", self.attention_boost_decay) or self.attention_boost_decay), 0.0, 1.0)
        if "attention_boost_max_extra_raw_budget" in payload:
            self.attention_boost_max_extra_raw_budget = max(0, int(payload.get("attention_boost_max_extra_raw_budget", self.attention_boost_max_extra_raw_budget) or self.attention_boost_max_extra_raw_budget))
        if "attention_boost_max_extra_focus_budget" in payload:
            self.attention_boost_max_extra_focus_budget = max(0, int(payload.get("attention_boost_max_extra_focus_budget", self.attention_boost_max_extra_focus_budget) or self.attention_boost_max_extra_focus_budget))
        if "attention_boost_min_radius_scale" in payload:
            self.attention_boost_min_radius_scale = _clamp(float(payload.get("attention_boost_min_radius_scale", self.attention_boost_min_radius_scale) or self.attention_boost_min_radius_scale), 0.05, 1.0)
        if "attention_boost_edge_gain" in payload:
            self.attention_boost_edge_gain = max(0.0, float(payload.get("attention_boost_edge_gain", self.attention_boost_edge_gain) or self.attention_boost_edge_gain))
        if "attention_boost_gaze_sigma_scale" in payload:
            self.attention_boost_gaze_sigma_scale = _clamp(float(payload.get("attention_boost_gaze_sigma_scale", self.attention_boost_gaze_sigma_scale) or self.attention_boost_gaze_sigma_scale), 0.05, 2.0)
        if "dynamic_track_window" in payload:
            self.dynamic_track_window = max(2, int(payload.get("dynamic_track_window", self.dynamic_track_window) or self.dynamic_track_window))
        if "dynamic_candidate_limit_background" in payload:
            self.dynamic_candidate_limit_background = max(2, int(payload.get("dynamic_candidate_limit_background", self.dynamic_candidate_limit_background) or self.dynamic_candidate_limit_background))
        if "dynamic_candidate_limit_focus" in payload:
            self.dynamic_candidate_limit_focus = max(self.dynamic_candidate_limit_background, int(payload.get("dynamic_candidate_limit_focus", self.dynamic_candidate_limit_focus) or self.dynamic_candidate_limit_focus))
        if "dynamic_track_limit" in payload:
            self.dynamic_track_limit = max(4, int(payload.get("dynamic_track_limit", self.dynamic_track_limit) or self.dynamic_track_limit))
        if "dynamic_summary_limit" in payload:
            self.dynamic_summary_limit = max(1, int(payload.get("dynamic_summary_limit", self.dynamic_summary_limit) or self.dynamic_summary_limit))
        if "dynamic_match_threshold" in payload:
            self.dynamic_match_threshold = _clamp(float(payload.get("dynamic_match_threshold", self.dynamic_match_threshold) or self.dynamic_match_threshold), 0.05, 0.95)
        if "dynamic_track_forget_ticks" in payload:
            self.dynamic_track_forget_ticks = max(1, int(payload.get("dynamic_track_forget_ticks", self.dynamic_track_forget_ticks) or self.dynamic_track_forget_ticks))
        if "export_preview_image" in payload:
            self.export_preview_image = bool(payload.get("export_preview_image", self.export_preview_image))
        boost = payload.get("attention_boost", {}) or {}
        if isinstance(boost, dict):
            self._attention_boost = {
                "active": bool(boost.get("active", False)),
                "strength": float(boost.get("strength", 0.0) or 0.0),
                "ticks_left": max(0, int(boost.get("ticks_left", 0) or 0)),
                "target_gaze": dict(boost.get("target_gaze", {}) or {"x": self.gaze_center[0], "y": self.gaze_center[1]}),
                "source_action": str(boost.get("source_action", "") or ""),
                "raw_budget_bonus": max(0, int(boost.get("raw_budget_bonus", 0) or 0)),
                "focus_budget_bonus": max(0, int(boost.get("focus_budget_bonus", 0) or 0)),
                "radius_scale": _clamp(float(boost.get("radius_scale", 1.0) or 1.0), 0.05, 1.0),
                "edge_gain": max(0.0, float(boost.get("edge_gain", 1.0) or 1.0)),
                "gaze_sigma_scale": _clamp(float(boost.get("gaze_sigma_scale", 1.0) or 1.0), 0.05, 2.0),
            }

    def _sample_original_resolution(
        self,
        *,
        image: Image.Image,
        sample_count: int,
        focus_priority_budget: int,
        memory_write_budget: int,
        source_type: str,
        sampling_profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float]]:
        width, height = image.size
        profile = dict(sampling_profile or {})
        positions = self._sample_candidate_positions(
            image=image,
            width=width,
            height=height,
            sample_count=sample_count,
            sampling_profile=profile,
        )
        raw_candidates: list[dict[str, Any]] = []
        motion_values: list[float] = []

        for pixel_x, pixel_y in positions:
            sample = self._build_sample(
                image=image,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                width=width,
                height=height,
                source_type=source_type,
                sampling_profile=profile,
                include_descriptor_tokens=False,
            )
            motion_values.append(float((sample.get("attributes", {}) or {}).get("motion", 0.0) or 0.0))
            raw_candidates.append(sample)

        raw_candidates.sort(
            key=lambda item: (
                -float((item.get("attributes", {}) or {}).get("raw_priority", 0.0) or 0.0),
                -float(item.get("energy", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            )
        )
        raw_samples = raw_candidates[:sample_count]

        focus_ranked = sorted(
            raw_samples,
            key=lambda item: (
                -float((item.get("attributes", {}) or {}).get("focus_priority", 0.0) or 0.0),
                -float(item.get("energy", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            ),
        )
        focus_samples = [
            self._to_memory_feature_sample(
                self._ensure_visual_descriptors(image=image, item=item),
                sample_reason="focus_memory",
            )
            for item in focus_ranked[:focus_priority_budget]
        ]

        selected_map: dict[str, dict[str, Any]] = {}
        for item in focus_samples:
            selected_map[str(item.get("sa_label", "") or "")] = item
        for item in raw_samples:
            memory_item = self._to_memory_feature_sample(
                self._ensure_visual_descriptors(image=image, item=item),
                sample_reason=str((item.get("attributes", {}) or {}).get("sample_reason", "raw") or "raw"),
            )
            label = str(memory_item.get("sa_label", "") or "")
            if label not in selected_map:
                selected_map[label] = memory_item
            if len(selected_map) >= max(1, int(memory_write_budget)):
                break
        selected_samples = list(selected_map.values())
        selected_samples.sort(key=lambda item: (-float(item.get("energy", 0.0) or 0.0), item.get("position", 0)))

        next_prev: dict[tuple[int, int], dict[str, float]] = {}
        next_selected_counts: dict[tuple[int, int], int] = {}
        selected_keys = {
            (int((item.get("coords", {}) or {}).get("pixel_x", -1) or -1), int((item.get("coords", {}) or {}).get("pixel_y", -1) or -1))
            for item in raw_samples[: max(1, int(memory_write_budget))]
        }
        for item in raw_samples:
            coords = dict(item.get("coords", {}) or {})
            key = (int(coords.get("pixel_x", -1) or -1), int(coords.get("pixel_y", -1) or -1))
            attrs = dict(item.get("attributes", {}) or {})
            next_prev[key] = {
                "brightness": float(attrs.get("brightness", 0.0) or 0.0),
                "avg_r": float(attrs.get("avg_r", 0.0) or 0.0),
                "avg_g": float(attrs.get("avg_g", 0.0) or 0.0),
                "avg_b": float(attrs.get("avg_b", 0.0) or 0.0),
            }
            prior_hits = int(self._recent_selected_counts.get(key, 0) or 0)
            next_selected_counts[key] = min(12, prior_hits + 1) if key in selected_keys else max(0, prior_hits - 1)
        self._prev_raw_samples = next_prev
        self._recent_selected_counts = next_selected_counts
        return raw_samples, selected_samples, focus_samples, motion_values

    def _sample_candidate_positions(
        self,
        *,
        image: Image.Image,
        width: int,
        height: int,
        sample_count: int,
        sampling_profile: dict[str, Any] | None = None,
    ) -> list[tuple[int, int]]:
        total_pixels = max(1, width * height)
        candidate_multiplier = 1.8
        if sample_count >= 64:
            candidate_multiplier = 1.35
        elif sample_count >= 32:
            candidate_multiplier = 1.5
        target_candidates = min(total_pixels, max(sample_count, int(sample_count * candidate_multiplier)))
        profile = dict(sampling_profile or {})
        rng_seed = (
            int(self._sensor_tick) * 1000003
            + width * 9176
            + height * 1361
            + int(self.gaze_center[0] * 1000.0) * 239
            + int(self.gaze_center[1] * 1000.0) * 521
        )
        rng = random.Random(rng_seed)
        positions: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def push(x: int, y: int) -> None:
            xx = min(width - 1, max(0, int(x)))
            yy = min(height - 1, max(0, int(y)))
            key = (xx, yy)
            if key in seen:
                return
            seen.add(key)
            positions.append(key)

        grid_quota = max(1, int(target_candidates * 0.45))
        focus_quota = max(1, int(target_candidates * 0.40))
        edge_quota = max(1, int(target_candidates * 0.18))
        fixation_quota = max(0, min(len(self._fixation_buffer), target_candidates - grid_quota - focus_quota - edge_quota))

        grid_step = max(1.0, math.sqrt(total_pixels / max(1.0, float(grid_quota))))
        phase_x = rng.random() * grid_step
        phase_y = rng.random() * grid_step
        y = phase_y
        while y < height and len(positions) < grid_quota:
            x = phase_x
            while x < width and len(positions) < grid_quota:
                jitter_x = rng.uniform(-0.42, 0.42) * grid_step
                jitter_y = rng.uniform(-0.42, 0.42) * grid_step
                push(int(round(x + jitter_x)), int(round(y + jitter_y)))
                x += grid_step
            y += grid_step

        gaze_px = self.gaze_center[0] * max(0, width - 1)
        gaze_py = self.gaze_center[1] * max(0, height - 1)
        boost_target = dict(profile.get("boost", {}) or {}).get("target_gaze", {}) or {}
        if bool(dict(profile.get("boost", {}) or {}).get("active", False)):
            gaze_px = float(boost_target.get("x", self.gaze_center[0]) or self.gaze_center[0]) * max(0, width - 1)
            gaze_py = float(boost_target.get("y", self.gaze_center[1]) or self.gaze_center[1]) * max(0, height - 1)
        sigma = max(2.0, min(width, height) * 0.18 * float(profile.get("gaze_sigma_scale", 1.0) or 1.0) * float(profile.get("radius_scale", 1.0) or 1.0))
        for _ in range(focus_quota * 3):
            if len(positions) >= grid_quota + focus_quota:
                break
            radius = abs(rng.gauss(0.0, sigma))
            angle = rng.random() * math.pi * 2.0
            px = int(round(gaze_px + math.cos(angle) * radius))
            py = int(round(gaze_py + math.sin(angle) * radius))
            push(px, py)

        if edge_quota > 0:
            probes = max(edge_quota * 4, 32)
            edge_scores: list[tuple[float, int, int]] = []
            candidate_gain = max(0.0, float(profile.get("candidate_edge_gain", self.edge_candidate_gain) or self.edge_candidate_gain))
            for _ in range(probes):
                if bool(dict(profile.get("boost", {}) or {}).get("active", False)) and rng.random() < 0.65:
                    radius = abs(rng.gauss(0.0, sigma * 1.1))
                    angle = rng.random() * math.pi * 2.0
                    px = int(round(gaze_px + math.cos(angle) * radius))
                    py = int(round(gaze_py + math.sin(angle) * radius))
                else:
                    px = rng.randrange(width)
                    py = rng.randrange(height)
                contrast = self._edge_probe_score(image=image, pixel_x=px, pixel_y=py)
                dx = px / max(1.0, float(width - 1)) - self.gaze_center[0]
                dy = py / max(1.0, float(height - 1)) - self.gaze_center[1]
                gaze_bonus = _clamp(1.0 - math.sqrt(dx * dx + dy * dy) * 1.45, 0.0, 1.0)
                score = contrast * candidate_gain + gaze_bonus * 0.18 + rng.random() * 0.03
                edge_scores.append((score, px, py))
            edge_scores.sort(key=lambda item: -item[0])
            for _, px, py in edge_scores[:edge_quota]:
                push(px, py)

        if fixation_quota > 0:
            fixation_rows = sorted(
                self._fixation_buffer.items(),
                key=lambda item: (
                    -float((item[1] or {}).get("energy", 0.0) or 0.0),
                    -int((item[1] or {}).get("sample_hits", 0) or 0),
                    item[0],
                ),
            )
            for (pixel_x, pixel_y), _ in fixation_rows[:fixation_quota]:
                push(pixel_x, pixel_y)

        while len(positions) < target_candidates:
            push(rng.randrange(width), rng.randrange(height))
        return positions[:target_candidates]

    def _build_sample(
        self,
        *,
        image: Image.Image,
        pixel_x: int,
        pixel_y: int,
        width: int,
        height: int,
        source_type: str,
        sampling_profile: dict[str, Any] | None = None,
        include_descriptor_tokens: bool = True,
    ) -> dict[str, Any]:
        profile = dict(sampling_profile or {})
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        avg_r = int(stats.get("avg_r_255", 0) or 0)
        avg_g = int(stats.get("avg_g_255", 0) or 0)
        avg_b = int(stats.get("avg_b_255", 0) or 0)
        brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        norm_rgb = (avg_r / 255.0, avg_g / 255.0, avg_b / 255.0)
        hue, saturation, value = colorsys.rgb_to_hsv(*norm_rgb)
        local_contrast = _clamp(float(stats.get("local_contrast", 0.0) or 0.0), 0.0, 1.0)
        grad_x = float(stats.get("gradient_x", 0.0) or 0.0)
        grad_y = float(stats.get("gradient_y", 0.0) or 0.0)
        gradient_mag = _clamp(float(stats.get("gradient_mag", 0.0) or 0.0), 0.0, 1.0)
        gradient_dir = _round4((math.atan2(grad_y, grad_x) + math.pi) / (2.0 * math.pi)) if gradient_mag > 0.0 else 0.0
        edge_strength = _clamp(local_contrast * 0.55 + gradient_mag * 0.45, 0.0, 1.0)
        stroke_likeness = _clamp(edge_strength * (0.55 + (1.0 - saturation) * 0.45), 0.0, 1.0)
        shape_metrics = self._local_shape_metrics(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        endpoint_likeness = float(shape_metrics.get("endpoint_likeness", 0.0) or 0.0)
        corner_likeness = float(shape_metrics.get("corner_likeness", 0.0) or 0.0)
        opening_likeness = float(shape_metrics.get("opening_likeness", 0.0) or 0.0)
        closure_likeness = float(shape_metrics.get("closure_likeness", 0.0) or 0.0)
        arc_balance = float(shape_metrics.get("arc_balance", 0.0) or 0.0)
        straight_likeness = float(shape_metrics.get("straight_likeness", 0.0) or 0.0)
        curvilinear_likeness = float(shape_metrics.get("curvilinear_likeness", 0.0) or 0.0)
        angularity = float(shape_metrics.get("angularity", 0.0) or 0.0)
        roundness = float(shape_metrics.get("roundness", 0.0) or 0.0)
        local_symmetry = float(shape_metrics.get("local_symmetry", 0.0) or 0.0)
        opening_dir_x = float(shape_metrics.get("opening_dir_x", 0.0) or 0.0)
        opening_dir_y = float(shape_metrics.get("opening_dir_y", 0.0) or 0.0)
        opening_direction_strength = float(shape_metrics.get("opening_direction_strength", 0.0) or 0.0)
        symmetry_support = local_symmetry * (0.25 + closure_likeness * 0.45 + roundness * 0.30)
        structure_discriminability = _clamp(
            endpoint_likeness * 0.22
            + opening_likeness * 0.24
            + corner_likeness * 0.16
            + opening_direction_strength * 0.20
            + abs(straight_likeness - curvilinear_likeness) * 0.10
            + abs(angularity - roundness) * 0.08
            - closure_likeness * 0.08,
            0.0,
            1.2,
        )

        screen_x = pixel_x / max(1.0, float(width))
        screen_y = pixel_y / max(1.0, float(height))
        screen_w = 1.0 / max(1.0, float(width))
        screen_h = 1.0 / max(1.0, float(height))
        cx = _clamp(screen_x + screen_w * 0.5, 0.0, 1.0)
        cy = _clamp(screen_y + screen_h * 0.5, 0.0, 1.0)
        dx_from_gaze = cx - self.gaze_center[0]
        dy_from_gaze = cy - self.gaze_center[1]
        radial_from_gaze = math.sqrt(dx_from_gaze * dx_from_gaze + dy_from_gaze * dy_from_gaze)
        gaze_bonus = _clamp(1.0 - radial_from_gaze * 1.65, 0.0, 1.0)
        if bool(dict(profile.get("boost", {}) or {}).get("active", False)):
            target = dict(dict(profile.get("boost", {}) or {}).get("target_gaze", {}) or {})
            target_dx = cx - float(target.get("x", self.gaze_center[0]) or self.gaze_center[0])
            target_dy = cy - float(target.get("y", self.gaze_center[1]) or self.gaze_center[1])
            target_radius = math.sqrt(target_dx * target_dx + target_dy * target_dy)
            target_bonus = _clamp(1.0 - target_radius * (1.65 / max(0.05, float(profile.get("radius_scale", 1.0) or 1.0))), 0.0, 1.0)
            gaze_bonus = max(gaze_bonus, target_bonus)

        key = (pixel_x, pixel_y)
        prev_state = self._prev_raw_samples.get(key, {})
        prev_rgb = (
            float(prev_state.get("avg_r", norm_rgb[0]) or norm_rgb[0]),
            float(prev_state.get("avg_g", norm_rgb[1]) or norm_rgb[1]),
            float(prev_state.get("avg_b", norm_rgb[2]) or norm_rgb[2]),
        )
        motion = _clamp(_difference_score(norm_rgb, prev_rgb) * 2.4, 0.0, 1.0)
        fatigue_hits = int(self._recent_selected_counts.get(key, 0) or 0)
        fatigue_penalty = min(0.18, fatigue_hits * 0.03) if motion < 0.08 else 0.0
        edge_gain = max(0.0, float(profile.get("edge_gain", self.edge_priority_gain) or self.edge_priority_gain))
        edge_priority = _clamp(edge_strength * edge_gain, 0.0, 2.2)
        stroke_priority = _clamp(stroke_likeness * (1.0 + max(0.0, edge_gain - 1.0) * 0.65), 0.0, 2.2)
        structure_priority = _clamp(
            endpoint_likeness * 0.20
            + corner_likeness * 0.16
            + opening_likeness * 0.14
            + closure_likeness * 0.16
            + arc_balance * 0.10
            + straight_likeness * 0.18
            + curvilinear_likeness * 0.16
            + angularity * 0.14
            + roundness * 0.18
            + symmetry_support * 0.12,
            0.0,
            2.2,
        )

        raw_priority = (
            brightness / 255.0 * 0.08
            + gaze_bonus * 0.24
            + edge_priority * 0.30
            + stroke_priority * 0.18
            + structure_priority * 0.16
            + structure_discriminability * 0.10
            + motion * 0.10
            - fatigue_penalty * 0.55
        )
        focus_priority = (
            gaze_bonus * 0.28
            + stroke_priority * 0.20
            + edge_priority * 0.16
            + structure_priority * 0.16
            + structure_discriminability * 0.18
            + motion * 0.06
        )
        energy = _clamp(0.08 + raw_priority, 0.05, 1.25)

        sample_reason = "gaze_focus" if gaze_bonus >= 0.72 else "edge_probe" if edge_priority >= 0.62 else "explore"

        coords = {
            "pixel_x": int(pixel_x),
            "pixel_y": int(pixel_y),
            "x": _round4(screen_x),
            "y": _round4(screen_y),
            "cx": _round4(cx),
            "cy": _round4(cy),
            "screen_x": _round4(screen_x),
            "screen_y": _round4(screen_y),
            "screen_w": _round4(screen_w),
            "screen_h": _round4(screen_h),
            "dx_from_gaze": _round4(dx_from_gaze),
            "dy_from_gaze": _round4(dy_from_gaze),
            "dr_from_gaze": _round4(radial_from_gaze),
        }
        attrs = {
            "brightness": _round4(brightness / 255.0),
            "avg_r": _round4(norm_rgb[0]),
            "avg_g": _round4(norm_rgb[1]),
            "avg_b": _round4(norm_rgb[2]),
            "hue": _round4(hue),
            "saturation": _round4(saturation),
            "value": _round4(value),
            "gaze_bonus": _round4(gaze_bonus),
            "motion": _round4(motion),
            "fatigue_penalty": _round4(fatigue_penalty),
            "relative_center_bias": _round4(max(0.0, 1.0 - radial_from_gaze * 1.6)),
            "local_contrast": _round4(local_contrast),
            "gradient_mag": _round4(gradient_mag),
            "gradient_dir": _round4(gradient_dir),
            "edge_strength": _round4(edge_strength),
            "stroke_likeness": _round4(stroke_likeness),
            "edge_priority": _round4(edge_priority),
            "stroke_priority": _round4(stroke_priority),
            "endpoint_likeness": _round4(endpoint_likeness),
            "corner_likeness": _round4(corner_likeness),
            "opening_likeness": _round4(opening_likeness),
            "closure_likeness": _round4(closure_likeness),
            "arc_balance": _round4(arc_balance),
            "straight_likeness": _round4(straight_likeness),
            "curvilinear_likeness": _round4(curvilinear_likeness),
            "angularity": _round4(angularity),
            "roundness": _round4(roundness),
            "local_symmetry": _round4(local_symmetry),
            "opening_dir_x": _round4(opening_dir_x),
            "opening_dir_y": _round4(opening_dir_y),
            "opening_direction_strength": _round4(opening_direction_strength),
            "symmetry_support": _round4(symmetry_support),
            "structure_discriminability": _round4(structure_discriminability),
            "structure_priority": _round4(structure_priority),
            "raw_priority": _round4(raw_priority),
            "focus_priority": _round4(focus_priority),
            "sample_reason": sample_reason,
            "sample_role": "raw_state",
        }
        if include_descriptor_tokens:
            attrs["local_patch_signature"] = self._local_patch_signature(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
            attrs.update(self._shape_descriptor_tokens(image=image, pixel_x=pixel_x, pixel_y=pixel_y))
        return {
            "sa_label": f"vision::{pixel_y}_{pixel_x}",
            "display_text": f"视觉采样[{pixel_x},{pixel_y}]",
            "energy": _round4(energy),
            "position": int(pixel_y * max(1, width) + pixel_x),
            "source_type": source_type,
            "sa_kind": "visual_sparse_sample_unit",
            "coords": coords,
            "attributes": attrs,
            "channel": "vision",
        }

    def _ensure_visual_descriptors(self, *, image: Image.Image, item: dict[str, Any]) -> dict[str, Any]:
        attrs = dict(item.get("attributes", {}) or {})
        if str(attrs.get("proj_h_bin", "") or ""):
            return item
        coords = dict(item.get("coords", {}) or {})
        pixel_x = int(coords.get("pixel_x", -1) or -1)
        pixel_y = int(coords.get("pixel_y", -1) or -1)
        if pixel_x < 0 or pixel_y < 0:
            return item
        updated = dict(item)
        merged_attrs = dict(attrs)
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        if "shape_metrics" not in stats:
            self._local_shape_metrics(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        if "patch_signature" not in stats:
            self._local_patch_signature(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        if "shape_descriptor_tokens" not in stats:
            self._shape_descriptor_tokens(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        merged_attrs["local_patch_signature"] = str(stats.get("patch_signature", "") or "")
        merged_attrs.update(dict(stats.get("shape_descriptor_tokens", {}) or {}))
        updated["attributes"] = merged_attrs
        return updated

    def _clone_with_reason(self, item: dict[str, Any], sample_reason: str) -> dict[str, Any]:
        cloned = {
            **item,
            "coords": dict(item.get("coords", {}) or {}),
            "attributes": dict(item.get("attributes", {}) or {}),
        }
        cloned["attributes"]["sample_reason"] = str(sample_reason or "")
        return cloned

    def _to_memory_feature_sample(self, item: dict[str, Any], *, sample_reason: str) -> dict[str, Any]:
        cloned = self._clone_with_reason(item, sample_reason)
        attrs = dict(cloned.get("attributes", {}) or {})
        coords = dict(cloned.get("coords", {}) or {})
        contour_signature = str(attrs.get("hu_signature", "") or "")
        contour_radial = str(attrs.get("radial_signature", "") or "")
        if contour_signature or contour_radial:
            area_bin = int(max(0, min(9, math.floor(float(attrs.get("area_ratio", 0.0) or 0.0) * 20.0))))
            fill_bin = int(max(0, min(9, math.floor(float(attrs.get("bbox_fill", 0.0) or 0.0) * 10.0))))
            solidity_bin = int(max(0, min(9, math.floor(float(attrs.get("solidity", 0.0) or 0.0) * 10.0))))
            aspect_ratio = float(attrs.get("aspect_ratio", 1.0) or 1.0)
            aspect_bin = int(max(0, min(9, math.floor(min(1.9, aspect_ratio) / 1.9 * 10.0))))
            hole_count = int(max(0, min(3, int(attrs.get("hole_count", 0) or 0))))
            proj_h = str(attrs.get("proj_h_bin", "") or "0000")[:4]
            proj_v = str(attrs.get("proj_v_bin", "") or "0000")[:4]
            radial_bin = str(attrs.get("radial_bin", "") or "0000")[:4]
            quadrant_bin = str(attrs.get("quadrant_bin", "") or "0000")[:4]
            polarity = str(attrs.get("foreground_polarity", "bright") or "bright")[:6]
            feature_code = (
                f"contour_{contour_signature[:7]}_{contour_radial[:8]}"
                f"_a{area_bin}_f{fill_bin}_s{solidity_bin}_r{aspect_bin}_h{hole_count}"
                f"_ph{proj_h}_pv{proj_v}_rb{radial_bin}_qb{quadrant_bin}_p{polarity}"
            )
            cloned["sa_label"] = f"vision_mem::{feature_code}"
            cloned["display_text"] = f"视觉轮廓特征[{feature_code}]"
            cloned["sa_kind"] = "visual_contour_feature_unit"
            cloned["attributes"]["sample_role"] = "memory_feature"
            cloned["attributes"]["memory_feature_code"] = feature_code
            return cloned
        r_bin = int(max(0, min(9, math.floor(float(attrs.get("avg_r", 0.0) or 0.0) * 10.0))))
        g_bin = int(max(0, min(9, math.floor(float(attrs.get("avg_g", 0.0) or 0.0) * 10.0))))
        b_bin = int(max(0, min(9, math.floor(float(attrs.get("avg_b", 0.0) or 0.0) * 10.0))))
        edge_bin = int(max(0, min(9, math.floor(float(attrs.get("edge_strength", 0.0) or 0.0) * 10.0))))
        stroke_bin = int(max(0, min(9, math.floor(float(attrs.get("stroke_likeness", 0.0) or 0.0) * 10.0))))
        endpoint_bin = int(max(0, min(9, math.floor(float(attrs.get("endpoint_likeness", 0.0) or 0.0) * 10.0))))
        corner_bin = int(max(0, min(9, math.floor(float(attrs.get("corner_likeness", 0.0) or 0.0) * 10.0))))
        opening_bin = int(max(0, min(9, math.floor(float(attrs.get("opening_likeness", 0.0) or 0.0) * 10.0))))
        closure_bin = int(max(0, min(9, math.floor(float(attrs.get("closure_likeness", 0.0) or 0.0) * 10.0))))
        arc_bin = int(max(0, min(9, math.floor(float(attrs.get("arc_balance", 0.0) or 0.0) * 10.0))))
        symmetry_bin = int(max(0, min(2, math.floor(float(attrs.get("local_symmetry", 0.0) or 0.0) * 3.0))))
        discriminability_bin = int(max(0, min(3, math.floor(float(attrs.get("structure_discriminability", 0.0) or 0.0) * 4.0))))
        opening_dir_x = float(attrs.get("opening_dir_x", 0.0) or 0.0)
        opening_dir_y = float(attrs.get("opening_dir_y", 0.0) or 0.0)
        opening_direction_strength = float(attrs.get("opening_direction_strength", 0.0) or 0.0)
        shape_scores = {
            "l": float(attrs.get("straight_likeness", 0.0) or 0.0),
            "c": float(attrs.get("curvilinear_likeness", 0.0) or 0.0),
            "a": float(attrs.get("angularity", 0.0) or 0.0),
            "r": float(attrs.get("roundness", 0.0) or 0.0),
        }
        shape_family, shape_strength = max(shape_scores.items(), key=lambda item: (item[1], item[0]))
        shape_strength_bin = int(max(0, min(3, math.floor(shape_strength * 4.0))))
        if float(attrs.get("closure_likeness", 0.0) or 0.0) >= 0.62 and float(attrs.get("roundness", 0.0) or 0.0) >= 0.42:
            opening_tag = "cl"
        elif opening_direction_strength >= 0.14 and float(attrs.get("opening_likeness", 0.0) or 0.0) >= 0.08:
            if abs(opening_dir_x) >= abs(opening_dir_y):
                opening_tag = "or" if opening_dir_x > 0 else "ol"
            else:
                opening_tag = "od" if opening_dir_y > 0 else "ou"
        else:
            opening_tag = "ox"
        rel_x_bin = int(max(0, min(9, math.floor((float(coords.get("dx_from_gaze", 0.0) or 0.0) + 1.0) * 5.0))))
        rel_y_bin = int(max(0, min(9, math.floor((float(coords.get("dy_from_gaze", 0.0) or 0.0) + 1.0) * 5.0))))
        abs_x_bin = int(max(0, min(11, math.floor(float(coords.get("cx", 0.0) or 0.0) * 12.0))))
        abs_y_bin = int(max(0, min(11, math.floor(float(coords.get("cy", 0.0) or 0.0) * 12.0))))
        signature = str(attrs.get("local_patch_signature", "") or "")[:9]
        proj_h = str(attrs.get("proj_h_bin", "") or "0000")[:4]
        proj_v = str(attrs.get("proj_v_bin", "") or "0000")[:4]
        orient_bin = str(attrs.get("orient_hist_bin", "") or "0000")[:4]
        radial_bin = str(attrs.get("radial_hist_bin", "") or "0000")[:4]
        hole_bin = int(max(0, min(3, math.floor(float(attrs.get("hole_like", 0.0) or 0.0) * 4.0))))
        center_void_bin = int(max(0, min(3, math.floor(float(attrs.get("center_void", 0.0) or 0.0) * 4.0))))
        hsym_bin = int(max(0, min(3, math.floor(float(attrs.get("horizontal_symmetry", 0.0) or 0.0) * 4.0))))
        vsym_bin = int(max(0, min(3, math.floor(float(attrs.get("vertical_symmetry", 0.0) or 0.0) * 4.0))))
        feature_code = (
            f"s{signature}_rgb{r_bin}{g_bin}{b_bin}_e{edge_bin}_k{stroke_bin}"
            f"_n{endpoint_bin}_c{corner_bin}_o{opening_bin}_q{closure_bin}_u{arc_bin}"
            f"_f{shape_family}{shape_strength_bin}_g{opening_tag}_y{symmetry_bin}_d{discriminability_bin}"
            f"_rx{rel_x_bin}_ry{rel_y_bin}_ax{abs_x_bin}_ay{abs_y_bin}"
            f"_ph{proj_h}_pv{proj_v}_oh{orient_bin}_rh{radial_bin}"
            f"_hl{hole_bin}_cv{center_void_bin}_hs{hsym_bin}_vs{vsym_bin}"
        )
        cloned["sa_label"] = f"vision_mem::{feature_code}"
        cloned["display_text"] = f"视觉特征[{feature_code}]"
        cloned["sa_kind"] = "visual_focus_feature_unit"
        cloned["attributes"]["sample_role"] = "memory_feature"
        cloned["attributes"]["memory_feature_code"] = feature_code
        return cloned

    def _encode_preview_data_url(self, image: Image.Image) -> str:
        preview = image.copy()
        max_edge = 640
        if max(preview.size) > max_edge:
            preview.thumbnail((max_edge, max_edge))
        buf = BytesIO()
        preview.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def _encode_png_data_url(self, array: np.ndarray, *, mode: str | None = None) -> str:
        if array is None or getattr(array, "size", 0) <= 0:
            return ""
        if mode:
            image = Image.fromarray(array, mode=mode)
        else:
            image = Image.fromarray(array)
        buf = BytesIO()
        image.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def _binary_signature4(self, values: np.ndarray | list[float] | list[int]) -> str:
        if values is None:
            return "0000"
        arr = np.asarray(values, dtype=np.float32)
        if arr.size <= 0:
            return "0000"
        bins: list[str] = []
        for part in np.array_split(arr, 4):
            if part.size <= 0:
                bins.append("0")
                continue
            mean = float(np.mean(part))
            bins.append(str(int(max(0, min(3, math.floor(_clamp(mean, 0.0, 0.9999) * 4.0))))))
        return "".join((bins + ["0", "0", "0", "0"])[:4])

    def _radial_signature8(self, contour: np.ndarray, *, center_x: float, center_y: float) -> str:
        if contour is None or contour.size <= 0:
            return "00000000"
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.size <= 0:
            return "00000000"
        dx = points[:, 0] - float(center_x)
        dy = points[:, 1] - float(center_y)
        radii = np.sqrt(dx * dx + dy * dy)
        max_radius = float(np.max(radii)) if radii.size > 0 else 0.0
        if max_radius <= 1e-6:
            return "00000000"
        angles = (np.arctan2(dy, dx) + math.pi) / (2.0 * math.pi)
        tokens: list[str] = []
        for bucket in range(8):
            lo = bucket / 8.0
            hi = (bucket + 1) / 8.0
            mask = (angles >= lo) & (angles < hi)
            if not np.any(mask):
                tokens.append("0")
                continue
            mean_r = float(np.mean(radii[mask])) / max_radius
            tokens.append(str(int(max(0, min(3, math.floor(_clamp(mean_r, 0.0, 0.9999) * 4.0))))))
        return "".join(tokens[:8])

    def _hu_signature(self, hu_values: np.ndarray) -> str:
        if hu_values is None or hu_values.size <= 0:
            return "0000000"
        flat = np.asarray(hu_values, dtype=np.float64).reshape(-1)
        tokens: list[str] = []
        for value in flat[:7]:
            if abs(float(value)) <= 1e-12:
                tokens.append("0")
                continue
            magnitude = min(15.0, max(0.0, -math.log10(abs(float(value)))))
            tokens.append(hex(int(round(magnitude)))[2:])
        return "".join((tokens + ["0"] * 7)[:7])

    def _rgb_signature3(self, rgb_values: np.ndarray | list[float] | tuple[float, float, float]) -> str:
        arr = np.asarray(rgb_values, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return "000"
        tokens: list[str] = []
        for value in arr[:3]:
            normalized = float(value)
            if normalized > 1.0:
                normalized /= 255.0
            tokens.append(str(int(max(0, min(3, math.floor(_clamp(normalized, 0.0, 0.9999) * 4.0))))))
        return "".join((tokens + ["0", "0", "0"])[:3])

    def _edge_contact_signature_array(self, binary: np.ndarray) -> str:
        if binary is None or binary.size <= 0 or binary.ndim != 2:
            return "0000"
        rows, cols = binary.shape
        if rows <= 0 or cols <= 0:
            return "0000"
        top = float(np.mean(binary[0:1, :])) if rows >= 1 else 0.0
        right = float(np.mean(binary[:, cols - 1 : cols])) if cols >= 1 else 0.0
        bottom = float(np.mean(binary[rows - 1 : rows, :])) if rows >= 1 else 0.0
        left = float(np.mean(binary[:, 0:1])) if cols >= 1 else 0.0
        return self._binary_signature4([top, right, bottom, left])

    def _bbox_signature(self, *, x: int, y: int, w: int, h: int, width: int, height: int) -> str:
        if width <= 0 or height <= 0:
            return "x0_y0_w0_h0"
        return (
            f"x{int(max(0, min(3, math.floor(_clamp(float(x) / max(1.0, float(width)), 0.0, 0.9999) * 4.0))))}"
            f"_y{int(max(0, min(3, math.floor(_clamp(float(y) / max(1.0, float(height)), 0.0, 0.9999) * 4.0))))}"
            f"_w{int(max(0, min(3, math.floor(_clamp(float(w) / max(1.0, float(width)), 0.0, 0.9999) * 4.0))))}"
            f"_h{int(max(0, min(3, math.floor(_clamp(float(h) / max(1.0, float(height)), 0.0, 0.9999) * 4.0))))}"
        )

    def _build_contour_bundle(self, *, image: Image.Image, source_type: str) -> dict[str, Any]:
        width, height = image.size
        if width <= 0 or height <= 0:
            return {
                "enabled": False,
                "reason": "empty_image",
                "mask_data_url": "",
                "outline_data_url": "",
                "silhouette_data_url": "",
                "foreground_data_url": "",
                "composite_data_url": "",
                "luma_edges_data_url": "",
                "color_edges_data_url": "",
                "components": [],
                "summary": {},
            }
        if cv2 is None:
            return {
                "enabled": False,
                "reason": "opencv_unavailable",
                "mask_data_url": "",
                "outline_data_url": "",
                "silhouette_data_url": "",
                "foreground_data_url": "",
                "composite_data_url": "",
                "luma_edges_data_url": "",
                "color_edges_data_url": "",
                "components": [],
                "summary": {},
            }
        rgb = self._rgb_array
        if rgb is None or rgb.size <= 0:
            rgb = np.asarray(image, dtype=np.uint8)
        gray = self._gray_array
        if gray is None or gray.size <= 0:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        border = np.concatenate((rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]), axis=0).astype(np.float32)
        bg_rgb = np.median(border, axis=0).astype(np.float32)
        bg_luma = float(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_rgb[None, None, :], axis=2) / math.sqrt(255.0 * 255.0 * 3.0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        abs_luma = np.abs(gray.astype(np.float32) - bg_luma) / 255.0
        salience = np.clip(diff * 0.60 + saturation * 0.24 + abs_luma * 0.16, 0.0, 1.0)
        sal_u8 = np.clip(salience * 255.0, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(sal_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        med = float(np.median(gray_u8))
        low = int(max(0.0, 0.66 * med))
        high = int(min(255.0, 1.33 * med + 8.0))
        edges = cv2.Canny(gray_u8, low, high)
        color_edges = np.zeros((height, width), dtype=np.uint8)
        for channel_index in range(3):
            color_edges = cv2.bitwise_or(color_edges, cv2.Canny(rgb[:, :, channel_index], low, high))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.bitwise_or(mask, cv2.dilate(edges, kernel3, iterations=1))
        mask = cv2.bitwise_or(mask, cv2.dilate(color_edges, kernel3, iterations=1))

        min_area = max(24, int(width * height * 0.0025))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean_mask = np.zeros((height, width), dtype=np.uint8)
        component_specs: list[dict[str, Any]] = []
        for index in range(1, int(num_labels)):
            x, y, comp_w, comp_h, area = [int(item) for item in stats[index].tolist()]
            touches_border = bool(x <= 0 or y <= 0 or (x + comp_w) >= width or (y + comp_h) >= height)
            if area < min_area:
                continue
            if touches_border and area < int(width * height * 0.20):
                continue
            clean_mask[labels == index] = 255
            component_specs.append(
                {
                    "label_index": int(index),
                    "x": x,
                    "y": y,
                    "w": comp_w,
                    "h": comp_h,
                    "area": int(area),
                    "touches_border": touches_border,
                }
            )
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel3, iterations=1)

        contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            hierarchy = np.zeros((1, 0, 4), dtype=np.int32)
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        outline = np.zeros((height, width, 4), dtype=np.uint8)
        silhouette = np.zeros((height, width, 4), dtype=np.uint8)
        foreground = np.zeros((height, width, 4), dtype=np.uint8)
        composite = np.zeros((height, width, 4), dtype=np.uint8)
        luma_edge_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        color_edge_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        contour_components: list[dict[str, Any]] = []

        fill_mask = np.zeros((height, width), dtype=np.uint8)
        for contour_index, contour in enumerate(top_contours[:6]):
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            x, y, comp_w, comp_h = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else area
            area_ratio = area / max(1.0, float(width * height))
            bbox_area = max(1, comp_w * comp_h)
            bbox_fill = area / max(1.0, float(bbox_area))
            aspect_ratio = comp_w / max(1.0, float(comp_h))
            extent = area / max(1.0, float(comp_w * comp_h))
            solidity = area / max(1.0, hull_area) if hull_area > 1e-6 else 0.0
            roundness = (4.0 * math.pi * area / max(1e-6, perimeter * perimeter)) if perimeter > 1e-6 else 0.0
            moments = cv2.moments(contour)
            if abs(float(moments.get("m00", 0.0) or 0.0)) > 1e-6:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float(x + comp_w * 0.5)
                cy = float(y + comp_h * 0.5)
            center_x = _clamp(cx / max(1.0, float(width - 1)), 0.0, 1.0)
            center_y = _clamp(cy / max(1.0, float(height - 1)), 0.0, 1.0)

            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
            holes = 0
            if hierarchy.shape[1] > 0:
                hierarchy_rows = hierarchy[0]
                current_hole = hierarchy_rows[contour_index][2] if contour_index < len(hierarchy_rows) else -1
                while int(current_hole) >= 0:
                    holes += 1
                    next_row = hierarchy_rows[int(current_hole)]
                    current_hole = int(next_row[0])
            ys, xs = np.where(component_mask > 0)
            if xs.size > 0 and ys.size > 0:
                colors = rgb[ys, xs, :].astype(np.float32)
                mean_rgb = np.mean(colors, axis=0)
            else:
                mean_rgb = bg_rgb
            component_binary = (component_mask[y : y + comp_h, x : x + comp_w] > 0).astype(np.float32)
            proj_h = self._binary_signature4(np.mean(component_mask[y : y + comp_h, x : x + comp_w] > 0, axis=1))
            proj_v = self._binary_signature4(np.mean(component_mask[y : y + comp_h, x : x + comp_w] > 0, axis=0))
            radial_signature = self._radial_signature8(contour, center_x=cx, center_y=cy)
            hu_signature = self._hu_signature(cv2.HuMoments(moments))
            radial_bin = self._binary_signature4([int(token) / 3.0 for token in radial_signature[:4]])
            quadrant_bin = self._binary_signature4(
                [
                    float(np.mean(component_mask[y : y + comp_h // 2, x : x + comp_w // 2] > 0)) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(component_mask[y : y + comp_h // 2, x + comp_w // 2 : x + comp_w] > 0)) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(component_mask[y + comp_h // 2 : y + comp_h, x : x + comp_w // 2] > 0)) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(component_mask[y + comp_h // 2 : y + comp_h, x + comp_w // 2 : x + comp_w] > 0)) if comp_h > 1 and comp_w > 1 else 0.0,
                ]
            )
            foreground_polarity = "bright" if float(np.mean(gray[component_mask > 0])) >= bg_luma else "dark"
            hole_like = _clamp(min(1.0, holes / 3.0) * 0.72 + max(0.0, 1.0 - bbox_fill) * 0.28, 0.0, 1.0)
            center_void = _clamp(max(0.0, 1.0 - bbox_fill), 0.0, 1.0)
            horizontal_symmetry = self._axis_symmetry_array(component_binary, axis="horizontal")
            vertical_symmetry = self._axis_symmetry_array(component_binary, axis="vertical")
            edge_contact_bin = self._edge_contact_signature_array(component_binary)
            bbox_signature = self._bbox_signature(x=x, y=y, w=comp_w, h=comp_h, width=width, height=height)
            rgb_signature = self._rgb_signature3(mean_rgb)
            contour_components.append(
                {
                    "component_id": f"contour_{contour_index}",
                    "rank": len(contour_components),
                    "area": _round4(area),
                    "area_ratio": _round4(area_ratio),
                    "perimeter": _round4(perimeter),
                    "bbox_fill": _round4(bbox_fill),
                    "extent": _round4(extent),
                    "solidity": _round4(solidity),
                    "roundness": _round4(_clamp(roundness, 0.0, 1.0)),
                    "aspect_ratio": _round4(aspect_ratio),
                    "hole_count": int(holes),
                    "hole_like": _round4(hole_like),
                    "center_void": _round4(center_void),
                    "horizontal_symmetry": _round4(horizontal_symmetry),
                    "vertical_symmetry": _round4(vertical_symmetry),
                    "proj_h_bin": proj_h,
                    "proj_v_bin": proj_v,
                    "radial_signature": radial_signature,
                    "radial_bin": radial_bin,
                    "quadrant_bin": quadrant_bin,
                    "edge_contact_bin": edge_contact_bin,
                    "bbox_signature": bbox_signature,
                    "rgb_signature": rgb_signature,
                    "hu_signature": hu_signature,
                    "foreground_polarity": foreground_polarity,
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(comp_w),
                        "h": int(comp_h),
                    },
                    "coords": {
                        "cx": _round4(center_x),
                        "cy": _round4(center_y),
                        "screen_x": _round4(x / max(1.0, float(width))),
                        "screen_y": _round4(y / max(1.0, float(height))),
                        "screen_w": _round4(comp_w / max(1.0, float(width))),
                        "screen_h": _round4(comp_h / max(1.0, float(height))),
                        "dx_from_gaze": _round4(center_x - self.gaze_center[0]),
                        "dy_from_gaze": _round4(center_y - self.gaze_center[1]),
                        "dr_from_gaze": _round4(math.sqrt((center_x - self.gaze_center[0]) ** 2 + (center_y - self.gaze_center[1]) ** 2)),
                    },
                    "mean_rgb": {
                        "r": _round4(float(mean_rgb[0]) / 255.0),
                        "g": _round4(float(mean_rgb[1]) / 255.0),
                        "b": _round4(float(mean_rgb[2]) / 255.0),
                    },
                }
            )
            cv2.drawContours(fill_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if contour_components:
            cv2.drawContours(outline, top_contours[: min(6, len(contour_components))], -1, (255, 247, 236, 255), thickness=2)
            silhouette[:, :, :3] = np.where(fill_mask[:, :, None] > 0, np.array([224, 236, 248], dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
            silhouette[:, :, 3] = np.where(fill_mask > 0, 168, 0).astype(np.uint8)
            edge_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(edge_mask, top_contours[: min(6, len(contour_components))], -1, 255, thickness=2)
            silhouette[edge_mask > 0, 0] = 255
            silhouette[edge_mask > 0, 1] = 248
            silhouette[edge_mask > 0, 2] = 236
            silhouette[edge_mask > 0, 3] = 255
            foreground[:, :, :3] = np.where(fill_mask[:, :, None] > 0, (rgb.astype(np.float32) * 0.95).astype(np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
            foreground[:, :, 3] = fill_mask
            luma_edge_rgba[:, :, :3] = np.where(edges[:, :, None] > 0, np.array([255, 245, 236], dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
            luma_edge_rgba[:, :, 3] = np.where(edges > 0, 255, 0).astype(np.uint8)
            color_edge_rgba[:, :, :3] = np.where(
                color_edges[:, :, None] > 0,
                np.clip(rgb.astype(np.float32) * 1.05, 0, 255).astype(np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )
            color_edge_rgba[:, :, 3] = np.where(color_edges > 0, 255, 0).astype(np.uint8)
            composite[:, :, :3] = np.where(
                fill_mask[:, :, None] > 0,
                np.clip(rgb.astype(np.float32) * 0.92 + 10.0, 0, 255).astype(np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
            )
            composite[:, :, 3] = np.where(fill_mask > 0, 176, 0).astype(np.uint8)
            composite[color_edges > 0, :3] = color_edge_rgba[color_edges > 0, :3]
            composite[color_edges > 0, 3] = 255
            composite[edges > 0, 0] = 255
            composite[edges > 0, 1] = 248
            composite[edges > 0, 2] = 236
            composite[edges > 0, 3] = 255
            composite[edge_mask > 0, 0] = 255
            composite[edge_mask > 0, 1] = 252
            composite[edge_mask > 0, 2] = 242
            composite[edge_mask > 0, 3] = 255

        summary = {
            "enabled": bool(contour_components),
            "component_count": len(contour_components),
            "primary_component_count": min(3, len(contour_components)),
            "foreground_coverage": _round4(float(np.count_nonzero(fill_mask)) / max(1.0, float(width * height))),
            "salience_mean": _round4(float(np.mean(salience))),
            "luma_edge_coverage": _round4(float(np.count_nonzero(edges)) / max(1.0, float(width * height))),
            "color_edge_coverage": _round4(float(np.count_nonzero(color_edges)) / max(1.0, float(width * height))),
            "background_rgb": {
                "r": _round4(float(bg_rgb[0]) / 255.0),
                "g": _round4(float(bg_rgb[1]) / 255.0),
                "b": _round4(float(bg_rgb[2]) / 255.0),
            },
        }
        return {
            "enabled": bool(contour_components),
            "reason": "ok" if contour_components else "no_component",
            "source_type": source_type,
            "mask_data_url": self._encode_png_data_url(clean_mask, mode="L"),
            "outline_data_url": self._encode_png_data_url(outline, mode="RGBA"),
            "silhouette_data_url": self._encode_png_data_url(silhouette, mode="RGBA"),
            "foreground_data_url": self._encode_png_data_url(foreground, mode="RGBA"),
            "composite_data_url": self._encode_png_data_url(composite, mode="RGBA"),
            "luma_edges_data_url": self._encode_png_data_url(luma_edge_rgba, mode="RGBA"),
            "color_edges_data_url": self._encode_png_data_url(color_edge_rgba, mode="RGBA"),
            "components": contour_components,
            "summary": summary,
        }

    def _empty_motion_contour_bundle(self, reason: str) -> dict[str, Any]:
        return {
            "motion_enabled": False,
            "motion_reason": str(reason or "unavailable"),
            "motion_mask_data_url": "",
            "motion_outline_data_url": "",
            "motion_composite_data_url": "",
            "motion_components": [],
            "motion_summary": {},
        }

    def _build_motion_contour_bundle(self, *, image: Image.Image) -> dict[str, Any]:
        if cv2 is None:
            return self._empty_motion_contour_bundle("opencv_unavailable")
        current_gray_arr = self._gray_array
        prev_gray = self._prev_frame_gray_u8
        current_rgb = self._rgb_array
        prev_rgb = self._prev_frame_rgb_u8
        if current_gray_arr is None or prev_gray is None or current_rgb is None or prev_rgb is None:
            return self._empty_motion_contour_bundle("no_previous_frame")
        if current_rgb.shape != prev_rgb.shape or current_gray_arr.shape != prev_gray.shape:
            return self._empty_motion_contour_bundle("shape_mismatch")

        height, width = current_rgb.shape[:2]
        gray_diff = np.abs(current_gray_arr.astype(np.float32) - prev_gray.astype(np.float32)) / 255.0
        rgb_delta = np.linalg.norm(current_rgb.astype(np.float32) - prev_rgb.astype(np.float32), axis=2) / math.sqrt(255.0 * 255.0 * 3.0)
        motion_signal = np.clip(rgb_delta * 0.62 + gray_diff * 0.38, 0.0, 1.0)
        motion_u8 = np.clip(motion_signal * 255.0, 0, 255).astype(np.uint8)
        if motion_u8.size <= 0:
            return self._empty_motion_contour_bundle("empty_signal")

        mean_signal = float(np.mean(motion_signal))
        max_signal = float(np.max(motion_signal))
        if max_signal <= 0.035:
            return self._empty_motion_contour_bundle("low_motion")

        _, mask = cv2.threshold(motion_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_threshold = int(max(8, min(72, round((mean_signal * 255.0) + 12.0))))
        mask = cv2.bitwise_or(mask, cv2.threshold(motion_u8, adaptive_threshold, 255, cv2.THRESH_BINARY)[1])
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.dilate(mask, kernel3, iterations=1)

        edge_low = int(max(4, min(96, round(np.median(motion_u8) * 0.66 + 4.0))))
        edge_high = int(max(edge_low + 4, min(255, round(np.median(motion_u8) * 1.42 + 16.0))))
        motion_edges = cv2.Canny(motion_u8, edge_low, edge_high)
        mask = cv2.bitwise_or(mask, cv2.dilate(motion_edges, kernel3, iterations=1))

        min_area = max(18, int(width * height * 0.0012))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean_mask = np.zeros((height, width), dtype=np.uint8)
        motion_components: list[dict[str, Any]] = []
        for index in range(1, int(num_labels)):
            x, y, comp_w, comp_h, area = [int(item) for item in stats[index].tolist()]
            if area < min_area:
                continue
            clean_mask[labels == index] = 255

        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            hierarchy = np.zeros((1, 0, 4), dtype=np.int32)
        fill_mask = np.zeros((height, width), dtype=np.uint8)
        outline = np.zeros((height, width, 4), dtype=np.uint8)
        composite = np.zeros((height, width, 4), dtype=np.uint8)

        ranked_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour_index, contour in enumerate(ranked_contours[:6]):
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            x, y, comp_w, comp_h = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else area
            bbox_area = max(1, comp_w * comp_h)
            area_ratio = area / max(1.0, float(width * height))
            bbox_fill = area / max(1.0, float(bbox_area))
            extent = area / max(1.0, float(bbox_area))
            solidity = area / max(1.0, hull_area) if hull_area > 1e-6 else 0.0
            roundness = (4.0 * math.pi * area / max(1e-6, perimeter * perimeter)) if perimeter > 1e-6 else 0.0
            moments = cv2.moments(contour)
            if abs(float(moments.get("m00", 0.0) or 0.0)) > 1e-6:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float(x + comp_w * 0.5)
                cy = float(y + comp_h * 0.5)
            center_x = _clamp(cx / max(1.0, float(width - 1)), 0.0, 1.0)
            center_y = _clamp(cy / max(1.0, float(height - 1)), 0.0, 1.0)
            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
            local_signal = motion_signal[y : y + comp_h, x : x + comp_w]
            local_binary = (component_mask[y : y + comp_h, x : x + comp_w] > 0).astype(np.float32)
            local_strength = float(np.mean(local_signal[local_binary > 0])) if np.count_nonzero(local_binary) > 0 else 0.0
            local_peak = float(np.max(local_signal[local_binary > 0])) if np.count_nonzero(local_binary) > 0 else 0.0
            holes = 0
            if hierarchy.shape[1] > 0:
                hierarchy_rows = hierarchy[0]
                current_hole = hierarchy_rows[contour_index][2] if contour_index < len(hierarchy_rows) else -1
                while int(current_hole) >= 0:
                    holes += 1
                    next_row = hierarchy_rows[int(current_hole)]
                    current_hole = int(next_row[0])
            proj_h = self._binary_signature4(np.mean(local_binary, axis=1))
            proj_v = self._binary_signature4(np.mean(local_binary, axis=0))
            radial_signature = self._radial_signature8(contour, center_x=cx, center_y=cy)
            hu_signature = self._hu_signature(cv2.HuMoments(moments))
            radial_bin = self._binary_signature4([int(token) / 3.0 for token in radial_signature[:4]])
            quadrant_bin = self._binary_signature4(
                [
                    float(np.mean(local_binary[: max(1, comp_h // 2), : max(1, comp_w // 2)])) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(local_binary[: max(1, comp_h // 2), comp_w // 2 :])) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(local_binary[comp_h // 2 :, : max(1, comp_w // 2)])) if comp_h > 1 and comp_w > 1 else 0.0,
                    float(np.mean(local_binary[comp_h // 2 :, comp_w // 2 :])) if comp_h > 1 and comp_w > 1 else 0.0,
                ]
            )
            horizontal_symmetry = self._axis_symmetry_array(local_binary, axis="horizontal")
            vertical_symmetry = self._axis_symmetry_array(local_binary, axis="vertical")
            edge_contact_bin = self._edge_contact_signature_array(local_binary)
            bbox_signature = self._bbox_signature(x=x, y=y, w=comp_w, h=comp_h, width=width, height=height)
            hole_like = _clamp(min(1.0, holes / 3.0) * 0.72 + max(0.0, 1.0 - bbox_fill) * 0.28, 0.0, 1.0)
            center_void = _clamp(max(0.0, 1.0 - bbox_fill), 0.0, 1.0)
            motion_components.append(
                {
                    "component_id": f"motion_{contour_index}",
                    "rank": len(motion_components),
                    "area_ratio": _round4(area_ratio),
                    "bbox_fill": _round4(bbox_fill),
                    "extent": _round4(extent),
                    "solidity": _round4(solidity),
                    "roundness": _round4(_clamp(roundness, 0.0, 1.0)),
                    "aspect_ratio": _round4(comp_w / max(1.0, float(comp_h))),
                    "hole_count": int(holes),
                    "hole_like": _round4(hole_like),
                    "center_void": _round4(center_void),
                    "horizontal_symmetry": _round4(horizontal_symmetry),
                    "vertical_symmetry": _round4(vertical_symmetry),
                    "proj_h_bin": proj_h,
                    "proj_v_bin": proj_v,
                    "radial_signature": radial_signature,
                    "radial_bin": radial_bin,
                    "quadrant_bin": quadrant_bin,
                    "edge_contact_bin": edge_contact_bin,
                    "bbox_signature": bbox_signature,
                    "hu_signature": hu_signature,
                    "motion_strength": _round4(local_strength),
                    "motion_peak": _round4(local_peak),
                    "coords": {
                        "cx": _round4(center_x),
                        "cy": _round4(center_y),
                        "screen_x": _round4(x / max(1.0, float(width))),
                        "screen_y": _round4(y / max(1.0, float(height))),
                        "screen_w": _round4(comp_w / max(1.0, float(width))),
                        "screen_h": _round4(comp_h / max(1.0, float(height))),
                        "dx_from_gaze": _round4(center_x - self.gaze_center[0]),
                        "dy_from_gaze": _round4(center_y - self.gaze_center[1]),
                        "dr_from_gaze": _round4(math.sqrt((center_x - self.gaze_center[0]) ** 2 + (center_y - self.gaze_center[1]) ** 2)),
                    },
                }
            )
            cv2.drawContours(fill_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if motion_components:
            cv2.drawContours(outline, ranked_contours[: min(6, len(motion_components))], -1, (166, 255, 224, 255), thickness=2)
            composite[:, :, :3] = np.where(fill_mask[:, :, None] > 0, np.array([72, 198, 166], dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
            composite[:, :, 3] = np.where(fill_mask > 0, 148, 0).astype(np.uint8)
            composite[motion_edges > 0, 0] = 214
            composite[motion_edges > 0, 1] = 255
            composite[motion_edges > 0, 2] = 236
            composite[motion_edges > 0, 3] = 255

        return {
            "motion_enabled": bool(motion_components),
            "motion_reason": "ok" if motion_components else "no_motion_component",
            "motion_mask_data_url": self._encode_png_data_url(clean_mask, mode="L"),
            "motion_outline_data_url": self._encode_png_data_url(outline, mode="RGBA"),
            "motion_composite_data_url": self._encode_png_data_url(composite, mode="RGBA"),
            "motion_components": motion_components,
            "motion_summary": {
                "component_count": len(motion_components),
                "motion_coverage": _round4(float(np.count_nonzero(fill_mask)) / max(1.0, float(width * height))),
                "motion_edge_coverage": _round4(float(np.count_nonzero(motion_edges)) / max(1.0, float(width * height))),
                "motion_signal_mean": _round4(mean_signal),
                "motion_signal_peak": _round4(max_signal),
            },
        }

    def _merge_contour_with_motion_bundle(
        self,
        *,
        contour_bundle: dict[str, Any],
        motion_bundle: dict[str, Any],
        source_type: str,
    ) -> dict[str, Any]:
        merged = copy.deepcopy(contour_bundle or {})
        merged["source_type"] = source_type
        merged.update(dict(motion_bundle or {}))
        return merged

    def _local_rgb(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> tuple[int, int, int]:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        return (
            int(stats.get("avg_r_255", 0) or 0),
            int(stats.get("avg_g_255", 0) or 0),
            int(stats.get("avg_b_255", 0) or 0),
        )

    def _local_contrast(self, *, image: Image.Image, pixel_x: int, pixel_y: int, center: tuple[float, float, float]) -> float:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        return _clamp(float(stats.get("local_contrast", 0.0) or 0.0), 0.0, 1.0)

    def _gradient(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        return (
            float(stats.get("gradient_x", 0.0) or 0.0),
            float(stats.get("gradient_y", 0.0) or 0.0),
        )

    def _edge_probe_score(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> float:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        local_contrast = float(stats.get("local_contrast", 0.0) or 0.0)
        gradient_mag = _clamp(float(stats.get("gradient_mag", 0.0) or 0.0), 0.0, 1.0)
        return _clamp(local_contrast * 0.52 + gradient_mag * 0.48, 0.0, 1.0)

    def _pixel_rgb(self, *, image: Image.Image, x: int, y: int) -> tuple[int, int, int]:
        width, height = image.size
        xx = min(width - 1, max(0, int(x)))
        yy = min(height - 1, max(0, int(y)))
        if self._pixel_access is not None:
            return tuple(self._pixel_access[xx, yy])
        return tuple(image.getpixel((xx, yy)))

    def _pixel_brightness(self, *, image: Image.Image, x: int, y: int) -> float:
        width, height = image.size
        xx = min(width - 1, max(0, int(x)))
        yy = min(height - 1, max(0, int(y)))
        if self._gray_pixel_access is not None:
            return float(self._gray_pixel_access[xx, yy]) / 255.0
        rr, gg, bb = self._pixel_rgb(image=image, x=xx, y=yy)
        return (0.299 * rr + 0.587 * gg + 0.114 * bb) / 255.0

    def _extract_rgb_patch(self, *, image: Image.Image, pixel_x: int, pixel_y: int, radius: int) -> np.ndarray | None:
        rgb = self._rgb_array
        if rgb is None:
            return None
        height, width = rgb.shape[:2]
        if width <= 0 or height <= 0:
            return None
        x0 = max(0, int(pixel_x) - int(radius))
        x1 = min(width, int(pixel_x) + int(radius) + 1)
        y0 = max(0, int(pixel_y) - int(radius))
        y1 = min(height, int(pixel_y) + int(radius) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = rgb[y0:y1, x0:x1]
        target = radius * 2 + 1
        if patch.shape[0] == target and patch.shape[1] == target:
            return patch
        pad_top = max(0, radius - int(pixel_y))
        pad_left = max(0, radius - int(pixel_x))
        pad_bottom = max(0, int(pixel_y) + radius + 1 - height)
        pad_right = max(0, int(pixel_x) + radius + 1 - width)
        return np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")

    def _extract_gray_patch(self, *, image: Image.Image, pixel_x: int, pixel_y: int, radius: int) -> np.ndarray | None:
        gray = self._gray_array
        if gray is None:
            return None
        height, width = gray.shape[:2]
        if width <= 0 or height <= 0:
            return None
        x0 = max(0, int(pixel_x) - int(radius))
        x1 = min(width, int(pixel_x) + int(radius) + 1)
        y0 = max(0, int(pixel_y) - int(radius))
        y1 = min(height, int(pixel_y) + int(radius) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = gray[y0:y1, x0:x1]
        target = radius * 2 + 1
        if patch.shape[0] == target and patch.shape[1] == target:
            return patch
        pad_top = max(0, radius - int(pixel_y))
        pad_left = max(0, radius - int(pixel_x))
        pad_bottom = max(0, int(pixel_y) + radius + 1 - height)
        pad_right = max(0, int(pixel_x) + radius + 1 - width)
        return np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

    def _patch_stats(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> dict[str, Any]:
        key = (int(pixel_x), int(pixel_y))
        cached = self._local_patch_cache.get(key)
        if cached is not None:
            return cached
        rgb_patch = self._extract_rgb_patch(image=image, pixel_x=pixel_x, pixel_y=pixel_y, radius=1)
        gray_patch = self._extract_gray_patch(image=image, pixel_x=pixel_x, pixel_y=pixel_y, radius=1)
        if rgb_patch is not None and gray_patch is not None and rgb_patch.shape[:2] == (3, 3) and gray_patch.shape == (3, 3):
            rgb_patch_f = rgb_patch.astype(np.float32)
            avg_rgb = np.mean(rgb_patch_f, axis=(0, 1))
            avg_r_255 = int(round(float(avg_rgb[0])))
            avg_g_255 = int(round(float(avg_rgb[1])))
            avg_b_255 = int(round(float(avg_rgb[2])))
            center = avg_rgb / 255.0
            neighbors = np.delete((rgb_patch_f / 255.0).reshape(-1, 3), 4, axis=0)
            if neighbors.size > 0:
                diffs = neighbors - center
                norms = np.sqrt(np.sum(diffs * diffs, axis=1)) / math.sqrt(3.0)
                local_contrast = _clamp(float(np.mean(norms)), 0.0, 1.0)
            else:
                local_contrast = 0.0
            gray_norm = gray_patch.astype(np.float32) / 255.0
            gx = (
                gray_norm[0, 2]
                + 2.0 * gray_norm[1, 2]
                + gray_norm[2, 2]
                - gray_norm[0, 0]
                - 2.0 * gray_norm[1, 0]
                - gray_norm[2, 0]
            ) / 4.0
            gy = (
                gray_norm[2, 0]
                + 2.0 * gray_norm[2, 1]
                + gray_norm[2, 2]
                - gray_norm[0, 0]
                - 2.0 * gray_norm[0, 1]
                - gray_norm[0, 2]
            ) / 4.0
        else:
            width, height = image.size
            brightness_grid: dict[tuple[int, int], float] = {}
            rgb_grid: dict[tuple[int, int], tuple[int, int, int]] = {}

            def rgb_at(x: int, y: int) -> tuple[int, int, int]:
                xx = min(width - 1, max(0, int(x)))
                yy = min(height - 1, max(0, int(y)))
                pos = (xx, yy)
                cached_rgb = rgb_grid.get(pos)
                if cached_rgb is not None:
                    return cached_rgb
                value = self._pixel_rgb(image=image, x=xx, y=yy)
                rgb_grid[pos] = value
                return value

            def bright_at(x: int, y: int) -> float:
                xx = min(width - 1, max(0, int(x)))
                yy = min(height - 1, max(0, int(y)))
                pos = (xx, yy)
                cached_b = brightness_grid.get(pos)
                if cached_b is not None:
                    return cached_b
                rr, gg, bb = rgb_at(xx, yy)
                value = (0.299 * rr + 0.587 * gg + 0.114 * bb) / 255.0
                brightness_grid[pos] = value
                return value

            total_r = 0
            total_g = 0
            total_b = 0
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    rr, gg, bb = rgb_at(pixel_x + dx, pixel_y + dy)
                    total_r += int(rr)
                    total_g += int(gg)
                    total_b += int(bb)
                    count += 1
            avg_r_255 = int(round(total_r / max(1, count)))
            avg_g_255 = int(round(total_g / max(1, count)))
            avg_b_255 = int(round(total_b / max(1, count)))
            center = (avg_r_255 / 255.0, avg_g_255 / 255.0, avg_b_255 / 255.0)

            total_contrast = 0.0
            contrast_count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    rr, gg, bb = rgb_at(pixel_x + dx, pixel_y + dy)
                    total_contrast += _difference_score(center, (rr / 255.0, gg / 255.0, bb / 255.0))
                    contrast_count += 1
            local_contrast = _clamp(total_contrast / max(1, contrast_count), 0.0, 1.0)

            gx = (
                bright_at(pixel_x + 1, pixel_y - 1)
                + 2.0 * bright_at(pixel_x + 1, pixel_y)
                + bright_at(pixel_x + 1, pixel_y + 1)
                - bright_at(pixel_x - 1, pixel_y - 1)
                - 2.0 * bright_at(pixel_x - 1, pixel_y)
                - bright_at(pixel_x - 1, pixel_y + 1)
            ) / 4.0
            gy = (
                bright_at(pixel_x - 1, pixel_y + 1)
                + 2.0 * bright_at(pixel_x, pixel_y + 1)
                + bright_at(pixel_x + 1, pixel_y + 1)
                - bright_at(pixel_x - 1, pixel_y - 1)
                - 2.0 * bright_at(pixel_x, pixel_y - 1)
                - bright_at(pixel_x + 1, pixel_y - 1)
            ) / 4.0
        gradient_mag = _clamp(math.sqrt(gx * gx + gy * gy) / 1.2, 0.0, 1.0)
        cached = {
            "avg_r_255": avg_r_255,
            "avg_g_255": avg_g_255,
            "avg_b_255": avg_b_255,
            "local_contrast": _round4(local_contrast),
            "gradient_x": gx,
            "gradient_y": gy,
            "gradient_mag": _round4(gradient_mag),
        }
        self._local_patch_cache[key] = cached
        return cached

    def _local_shape_metrics(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> dict[str, float]:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        cached_metrics = stats.get("shape_metrics")
        if isinstance(cached_metrics, dict):
            return dict(cached_metrics)
        gray_patch = self._extract_gray_patch(image=image, pixel_x=pixel_x, pixel_y=pixel_y, radius=2)
        if gray_patch is not None and gray_patch.shape == (5, 5):
            gray_norm = gray_patch.astype(np.float32) / 255.0
            center_b = float(gray_norm[2, 2])
            ring_vals = [
                float(gray_norm[0, 2]),
                float(gray_norm[0, 4]),
                float(gray_norm[2, 4]),
                float(gray_norm[4, 4]),
                float(gray_norm[4, 2]),
                float(gray_norm[4, 0]),
                float(gray_norm[2, 0]),
                float(gray_norm[0, 0]),
            ]
        else:
            center_b = self._pixel_brightness(image=image, x=pixel_x, y=pixel_y)
            ring_vals = [
                self._pixel_brightness(image=image, x=pixel_x + 0, y=pixel_y - 2),
                self._pixel_brightness(image=image, x=pixel_x + 2, y=pixel_y - 2),
                self._pixel_brightness(image=image, x=pixel_x + 2, y=pixel_y + 0),
                self._pixel_brightness(image=image, x=pixel_x + 2, y=pixel_y + 2),
                self._pixel_brightness(image=image, x=pixel_x + 0, y=pixel_y + 2),
                self._pixel_brightness(image=image, x=pixel_x - 2, y=pixel_y + 2),
                self._pixel_brightness(image=image, x=pixel_x - 2, y=pixel_y + 0),
                self._pixel_brightness(image=image, x=pixel_x - 2, y=pixel_y - 2),
            ]
        ring_dirs = [
            (0, -2),
            (2, -2),
            (2, 0),
            (2, 2),
            (0, 2),
            (-2, 2),
            (-2, 0),
            (-2, -2),
        ]
        threshold = max(0.18, center_b * 0.72)
        strengths = [_clamp((value - threshold) / max(0.05, 1.0 - threshold), 0.0, 1.0) for value in ring_vals]
        active = [1 if value >= threshold else 0 for value in ring_vals]
        active_count = sum(active)

        transitions = 0
        active_runs = 0
        prev = active[-1]
        for current in active:
            if current != prev:
                transitions += 1
                if current == 1:
                    active_runs += 1
            prev = current

        cardinal_vals = [ring_vals[0], ring_vals[2], ring_vals[4], ring_vals[6]]
        diagonal_vals = [ring_vals[1], ring_vals[3], ring_vals[5], ring_vals[7]]
        cardinal_mean = sum(cardinal_vals) / max(1, len(cardinal_vals))
        diagonal_mean = sum(diagonal_vals) / max(1, len(diagonal_vals))
        closure_likeness = _clamp((active_count / 8.0) * max(0.0, 1.0 - transitions / 8.0), 0.0, 1.0)
        endpoint_likeness = _clamp((1.0 - active_count / 8.0) * (transitions / 4.0) * max(0.0, center_b), 0.0, 1.0)
        corner_likeness = _clamp(abs(cardinal_mean - diagonal_mean) * 1.8 + max(0.0, active_runs - 1) * 0.18, 0.0, 1.0)
        opening_likeness = _clamp((active_runs / 3.0) * max(0.0, 1.0 - closure_likeness) * max(0.0, active_count / 8.0), 0.0, 1.0)
        upper_arc = (ring_vals[7] + ring_vals[0] + ring_vals[1]) / 3.0
        lower_arc = (ring_vals[3] + ring_vals[4] + ring_vals[5]) / 3.0
        arc_balance = _clamp(1.0 - abs(upper_arc - lower_arc) * 1.6, 0.0, 1.0)
        opposite_pairs = [
            (strengths[0] + strengths[4]) / 2.0,
            (strengths[1] + strengths[5]) / 2.0,
            (strengths[2] + strengths[6]) / 2.0,
            (strengths[3] + strengths[7]) / 2.0,
        ]
        pair_sorted = sorted(opposite_pairs, reverse=True)
        max_pair = pair_sorted[0] if pair_sorted else 0.0
        second_pair = pair_sorted[1] if len(pair_sorted) > 1 else 0.0
        line_selectivity = _clamp(max_pair - second_pair, 0.0, 1.0)

        arc_triplets = [sum(strengths[i : i + 3]) for i in range(6)]
        arc_triplets.extend(
            [
                strengths[6] + strengths[7] + strengths[0],
                strengths[7] + strengths[0] + strengths[1],
            ]
        )
        max_arc = max(arc_triplets) / 3.0 if arc_triplets else 0.0

        reflection_axes = [
            [(1, 7), (2, 6), (3, 5)],
            [(0, 4), (1, 3), (7, 5)],
            [(0, 6), (1, 5), (2, 4)],
            [(0, 2), (7, 3), (6, 4)],
        ]
        axis_symmetries: list[float] = []
        for pairs in reflection_axes:
            if not pairs:
                axis_symmetries.append(0.0)
                continue
            axis_symmetries.append(
                _clamp(
                    1.0 - (sum(abs(strengths[left] - strengths[right]) for left, right in pairs) / float(len(pairs))),
                    0.0,
                    1.0,
                )
            )
        local_symmetry = max(axis_symmetries) if axis_symmetries else 0.0
        vec_x = 0.0
        vec_y = 0.0
        total_gap = 0.0
        for (dx, dy), strength in zip(ring_dirs, strengths):
            gap = max(0.0, 1.0 - strength)
            length = math.sqrt(float(dx * dx + dy * dy))
            if gap <= 0.0 or length <= 0.0:
                continue
            vec_x += (dx / length) * gap
            vec_y += (dy / length) * gap
            total_gap += gap
        opening_dir_x = 0.0
        opening_dir_y = 0.0
        opening_direction_strength = 0.0
        if total_gap > 1e-6:
            vec_x /= total_gap
            vec_y /= total_gap
            opening_direction_strength = _clamp(math.sqrt(vec_x * vec_x + vec_y * vec_y) * opening_likeness, 0.0, 1.0)
            if opening_direction_strength > 1e-6:
                norm = math.sqrt(vec_x * vec_x + vec_y * vec_y)
                if norm > 1e-6:
                    opening_dir_x = _clamp(vec_x / norm, -1.0, 1.0)
                    opening_dir_y = _clamp(vec_y / norm, -1.0, 1.0)

        straight_likeness = _clamp(
            max_pair * 0.62
            + line_selectivity * 0.46
            + max(0.0, 1.0 - transitions / 5.0) * 0.10
            - closure_likeness * 0.16,
            0.0,
            1.0,
        )
        angularity = _clamp(
            corner_likeness * 0.58
            + line_selectivity * 0.18
            + max(0.0, active_runs - 1) * 0.10
            + max(0.0, 1.0 - arc_balance) * 0.24,
            0.0,
            1.0,
        )
        curvilinear_likeness = _clamp(
            max_arc * 0.46
            + arc_balance * 0.24
            + closure_likeness * 0.12
            + local_symmetry * 0.08
            - straight_likeness * 0.18
            - angularity * 0.12,
            0.0,
            1.0,
        )
        roundness = _clamp(
            closure_likeness * 0.30
            + arc_balance * 0.22
            + curvilinear_likeness * 0.20
            + local_symmetry * 0.18
            + max(0.0, 1.0 - opening_likeness) * 0.10
            - angularity * 0.18,
            0.0,
            1.0,
        )
        metrics = {
            "endpoint_likeness": _round4(endpoint_likeness),
            "corner_likeness": _round4(corner_likeness),
            "opening_likeness": _round4(opening_likeness),
            "closure_likeness": _round4(closure_likeness),
            "arc_balance": _round4(arc_balance),
            "straight_likeness": _round4(straight_likeness),
            "curvilinear_likeness": _round4(curvilinear_likeness),
            "angularity": _round4(angularity),
            "roundness": _round4(roundness),
            "local_symmetry": _round4(local_symmetry),
            "opening_dir_x": _round4(opening_dir_x),
            "opening_dir_y": _round4(opening_dir_y),
            "opening_direction_strength": _round4(opening_direction_strength),
        }
        stats["shape_metrics"] = dict(metrics)
        return metrics

    def _local_patch_signature(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> str:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        cached_signature = stats.get("patch_signature")
        if isinstance(cached_signature, str) and cached_signature:
            return cached_signature
        gray_patch = self._extract_gray_patch(image=image, pixel_x=pixel_x, pixel_y=pixel_y, radius=1)
        if gray_patch is not None and gray_patch.shape == (3, 3):
            gray_norm = gray_patch.astype(np.float32) / 255.0
            buckets = [
                str(int(_clamp(math.floor(float(value) * 10.0), 0, 9)))
                for value in gray_norm.reshape(-1)
            ]
        else:
            buckets: list[str] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    brightness = self._pixel_brightness(image=image, x=pixel_x + dx, y=pixel_y + dy)
                    buckets.append(str(int(_clamp(math.floor(brightness * 10.0), 0, 9))))
        signature = "".join(buckets)
        stats["patch_signature"] = signature
        return signature

    def _shape_descriptor_tokens(self, *, image: Image.Image, pixel_x: int, pixel_y: int) -> dict[str, Any]:
        stats = self._patch_stats(image=image, pixel_x=pixel_x, pixel_y=pixel_y)
        cached_tokens = stats.get("shape_descriptor_tokens")
        if isinstance(cached_tokens, dict) and cached_tokens:
            return dict(cached_tokens)
        width, height = image.size
        radius = 4
        x0 = max(0, pixel_x - radius)
        x1 = min(width, pixel_x + radius + 1)
        y0 = max(0, pixel_y - radius)
        y1 = min(height, pixel_y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            tokens = {
                "proj_h_bin": "0000",
                "proj_v_bin": "0000",
                "orient_hist_bin": "0000",
                "radial_hist_bin": "0000",
                "horizontal_symmetry": 0.0,
                "vertical_symmetry": 0.0,
                "hole_like": 0.0,
                "center_void": 0.0,
            }
            stats["shape_descriptor_tokens"] = dict(tokens)
            return tokens

        if self._gray_array is not None:
            patch = self._gray_array[y0:y1, x0:x1] / 255.0
        else:
            patch = np.asarray(self._gray_image or image.convert("L"), dtype=np.float32)[y0:y1, x0:x1] / 255.0
        rows, cols = patch.shape if patch.ndim == 2 else (0, 0)
        if rows <= 0 or cols <= 0:
            tokens = {
                "proj_h_bin": "0000",
                "proj_v_bin": "0000",
                "orient_hist_bin": "0000",
                "radial_hist_bin": "0000",
                "horizontal_symmetry": 0.0,
                "vertical_symmetry": 0.0,
                "hole_like": 0.0,
                "center_void": 0.0,
            }
            stats["shape_descriptor_tokens"] = dict(tokens)
            return tokens

        binary = (patch >= 0.45).astype(np.float32)
        row_sums = binary.mean(axis=1)
        col_sums = binary.mean(axis=0)

        padded = np.pad(patch, 1, mode="edge")
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        gx = right - left
        gy = down - up
        mag = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)
        orient_acc = [0.0, 0.0, 0.0, 0.0]
        for bucket in range(4):
            lo = bucket / 4.0
            hi = (bucket + 1) / 4.0
            mask = (angle >= lo) & (angle < hi) & (mag > 1e-6)
            orient_acc[bucket] = float(np.sum(mag[mask]))

        yy, xx = np.indices((rows, cols))
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_r = float(np.max(rr)) or 1.0
        rr = rr / max_r
        radial_acc = [0.0, 0.0, 0.0, 0.0]
        for bucket in range(4):
            lo = bucket / 4.0
            hi = 1.01 if bucket == 3 else (bucket + 1) / 4.0
            mask = (rr >= lo) & (rr < hi)
            radial_acc[bucket] = float(np.sum(patch[mask]))

        orient_total = sum(orient_acc) or 1.0
        radial_total = sum(radial_acc) or 1.0
        orient_hist_bin = "".join(str(int(max(0, min(3, math.floor((value / orient_total) * 4.0))))) for value in orient_acc)
        radial_hist_bin = "".join(str(int(max(0, min(3, math.floor((value / radial_total) * 4.0))))) for value in radial_acc)

        horizontal_symmetry = 0.0
        if rows > 1:
            top = binary[: rows // 2, :]
            bottom = np.flipud(binary[-(rows // 2) :, :])
            if top.size and bottom.size:
                horizontal_symmetry = _clamp(1.0 - float(np.mean(np.abs(top - bottom))), 0.0, 1.0)
        vertical_symmetry = 0.0
        if cols > 1:
            left_half = binary[:, : cols // 2]
            right_half = np.fliplr(binary[:, -(cols // 2) :])
            if left_half.size and right_half.size:
                vertical_symmetry = _clamp(1.0 - float(np.mean(np.abs(left_half - right_half))), 0.0, 1.0)

        center_patch = binary[max(0, rows // 2 - 1) : min(rows, rows // 2 + 2), max(0, cols // 2 - 1) : min(cols, cols // 2 + 2)]
        center_void = _clamp(1.0 - float(np.mean(center_patch)) if center_patch.size else 0.0, 0.0, 1.0)
        edge_band = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1])) if rows > 0 and cols > 0 else np.zeros((0,), dtype=np.float32)
        edge_density = float(np.mean(edge_band)) if edge_band.size else 0.0
        hole_like = _clamp(center_void * edge_density, 0.0, 1.0)

        tokens = {
            "proj_h_bin": _four_bin_code(row_sums),
            "proj_v_bin": _four_bin_code(col_sums),
            "orient_hist_bin": orient_hist_bin,
            "radial_hist_bin": radial_hist_bin,
            "horizontal_symmetry": _round4(horizontal_symmetry),
            "vertical_symmetry": _round4(vertical_symmetry),
            "hole_like": _round4(hole_like),
            "center_void": _round4(center_void),
        }
        stats["shape_descriptor_tokens"] = dict(tokens)
        return tokens

    def _build_global_structure_samples(
        self,
        *,
        image: Image.Image,
        source_type: str,
        contour_bundle: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        width, height = image.size
        if width <= 0 or height <= 0:
            return []

        contour_bundle = dict(contour_bundle or {})
        contour_components = [dict(item) for item in (contour_bundle.get("components", []) or []) if isinstance(item, dict)]
        if contour_components:
            samples: list[dict[str, Any]] = []
            for index, component in enumerate(contour_components[:3]):
                coords = dict(component.get("coords", {}) or {})
                mean_rgb = dict(component.get("mean_rgb", {}) or {})
                attrs = {
                    "sample_role": "global_structure",
                    "global_feature_group": "contour_component",
                    "global_feature_code": f"global_contour::{component.get('hu_signature', '0000000')}::{component.get('radial_signature', '00000000')}",
                    "component_id": str(component.get("component_id", "") or f"contour_{index}"),
                    "hu_signature": str(component.get("hu_signature", "") or "0000000"),
                    "radial_signature": str(component.get("radial_signature", "") or "00000000"),
                    "proj_h_bin": str(component.get("proj_h_bin", "") or "0000"),
                    "proj_v_bin": str(component.get("proj_v_bin", "") or "0000"),
                    "radial_bin": str(component.get("radial_bin", "") or "0000"),
                    "quadrant_bin": str(component.get("quadrant_bin", "") or "0000"),
                    "edge_contact_bin": str(component.get("edge_contact_bin", "") or "0000"),
                    "bbox_signature": str(component.get("bbox_signature", "") or "x0_y0_w0_h0"),
                    "rgb_signature": str(component.get("rgb_signature", "") or "000"),
                    "foreground_polarity": str(component.get("foreground_polarity", "bright") or "bright"),
                    "area_ratio": _round4(float(component.get("area_ratio", 0.0) or 0.0)),
                    "bbox_fill": _round4(float(component.get("bbox_fill", 0.0) or 0.0)),
                    "extent": _round4(float(component.get("extent", 0.0) or 0.0)),
                    "solidity": _round4(float(component.get("solidity", 0.0) or 0.0)),
                    "roundness": _round4(float(component.get("roundness", 0.0) or 0.0)),
                    "aspect_ratio": _round4(float(component.get("aspect_ratio", 0.0) or 0.0)),
                    "hole_count": int(component.get("hole_count", 0) or 0),
                    "hole_like": _round4(float(component.get("hole_like", 0.0) or 0.0)),
                    "center_void": _round4(float(component.get("center_void", 0.0) or 0.0)),
                    "horizontal_symmetry": _round4(float(component.get("horizontal_symmetry", 0.0) or 0.0)),
                    "vertical_symmetry": _round4(float(component.get("vertical_symmetry", 0.0) or 0.0)),
                    "avg_r": _round4(float(mean_rgb.get("r", 0.0) or 0.0)),
                    "avg_g": _round4(float(mean_rgb.get("g", 0.0) or 0.0)),
                    "avg_b": _round4(float(mean_rgb.get("b", 0.0) or 0.0)),
                    "brightness": _round4(
                        0.299 * float(mean_rgb.get("r", 0.0) or 0.0)
                        + 0.587 * float(mean_rgb.get("g", 0.0) or 0.0)
                        + 0.114 * float(mean_rgb.get("b", 0.0) or 0.0)
                    ),
                    "edge_strength": _round4(min(1.0, 0.28 + float(component.get("solidity", 0.0) or 0.0) * 0.36 + float(component.get("roundness", 0.0) or 0.0) * 0.18)),
                    "stroke_likeness": _round4(min(1.0, 0.24 + float(component.get("extent", 0.0) or 0.0) * 0.38)),
                    "structure_discriminability": _round4(
                        _clamp(
                            0.28
                            + float(component.get("hole_like", 0.0) or 0.0) * 0.20
                            + abs(float(component.get("aspect_ratio", 0.0) or 0.0) - 1.0) * 0.16
                            + float(component.get("solidity", 0.0) or 0.0) * 0.18,
                            0.0,
                            1.0,
                        )
                    ),
                    "local_patch_signature": str(component.get("hu_signature", "") or "0000000")[:9],
                }
                samples.append(
                    {
                        "sa_label": f"vision_mem::global_contour::{component.get('hu_signature', '0000000')}::{component.get('radial_signature', '00000000')}::{index}",
                        "display_text": f"视觉轮廓[{index + 1}]",
                        "energy": _round4(_clamp(0.30 + float(component.get("area_ratio", 0.0) or 0.0) * 2.8, 0.18, 1.18)),
                        "position": int(width * height + index),
                        "source_type": source_type,
                        "sa_kind": "visual_global_feature_unit",
                        "channel": "vision",
                        "coords": coords,
                        "attributes": attrs,
                    }
                )
            if samples:
                return samples

        gray_array = self._gray_array
        if gray_array is None:
            gray_array = np.asarray(self._gray_image or image.convert("L"), dtype=np.float32)
        if gray_array is None or gray_array.size <= 0:
            return []

        gray_array = gray_array.astype(np.float32, copy=False)
        mean_value = float(np.mean(gray_array))
        std_value = float(np.std(gray_array))
        delta = max(10.0, min(96.0, 8.0 + std_value * 0.58))
        bright_mask = gray_array >= (mean_value + delta)
        dark_mask = gray_array <= (mean_value - delta)
        bright_count = int(np.count_nonzero(bright_mask))
        dark_count = int(np.count_nonzero(dark_mask))
        if bright_count <= 0 and dark_count <= 0:
            delta = max(6.0, min(72.0, 4.0 + std_value * 0.34))
            bright_mask = gray_array >= (mean_value + delta)
            dark_mask = gray_array <= (mean_value - delta)
            bright_count = int(np.count_nonzero(bright_mask))
            dark_count = int(np.count_nonzero(dark_mask))
        foreground_is_bright = True
        if bright_count > 0 and dark_count > 0:
            foreground_is_bright = bright_count <= dark_count
        elif dark_count > 0:
            foreground_is_bright = False
        elif bright_count > 0:
            foreground_is_bright = True

        if foreground_is_bright:
            binary = (gray_array >= (mean_value + delta)).astype(np.uint8)
        else:
            binary = (gray_array <= (mean_value - delta)).astype(np.uint8)
        active_count = int(np.count_nonzero(binary))
        if active_count <= 0:
            if foreground_is_bright:
                binary = (gray_array >= mean_value).astype(np.uint8)
            else:
                binary = (gray_array <= mean_value).astype(np.uint8)
            active_count = int(np.count_nonzero(binary))

        total_pixels = max(1, width * height)
        if active_count <= 0:
            return []

        row_sums_np = binary.sum(axis=1).astype(np.float32)
        col_sums_np = binary.sum(axis=0).astype(np.float32)
        active_rows = np.flatnonzero(row_sums_np)
        active_cols = np.flatnonzero(col_sums_np)
        if active_rows.size <= 0 or active_cols.size <= 0:
            return []

        min_y = int(active_rows[0])
        max_y = int(active_rows[-1])
        min_x = int(active_cols[0])
        max_x = int(active_cols[-1])
        bbox_w = max(1, max_x - min_x + 1)
        bbox_h = max(1, max_y - min_y + 1)
        bbox_area = max(1, bbox_w * bbox_h)
        bbox_cx = (min_x + max_x) * 0.5 / max(1.0, float(width - 1))
        bbox_cy = (min_y + max_y) * 0.5 / max(1.0, float(height - 1))
        ink_density = active_count / float(total_pixels)
        bbox_fill = active_count / float(bbox_area)
        aspect = bbox_w / max(1.0, float(bbox_h))

        row_densities = row_sums_np[min_y : max_y + 1] / max(1.0, float(width))
        col_densities = col_sums_np[min_x : max_x + 1] / max(1.0, float(height))
        proj_h = self._four_bin_signature_array(row_densities)
        proj_v = self._four_bin_signature_array(col_densities)

        center_margin_x = max(1, int(round(bbox_w * 0.18)))
        center_margin_y = max(1, int(round(bbox_h * 0.18)))
        center_x0 = min_x + center_margin_x
        center_x1 = max_x - center_margin_x + 1
        center_y0 = min_y + center_margin_y
        center_y1 = max_y - center_margin_y + 1
        if center_x1 > center_x0 and center_y1 > center_y0:
            center_patch = binary[center_y0:center_y1, center_x0:center_x1]
            center_void = _clamp(1.0 - float(np.mean(center_patch)), 0.0, 1.0)
        else:
            center_void = 0.0

        bbox_binary = binary[min_y : max_y + 1, min_x : max_x + 1]
        horizontal_symmetry = self._axis_symmetry_array(bbox_binary, axis="horizontal")
        vertical_symmetry = self._axis_symmetry_array(bbox_binary, axis="vertical")

        active_positions = np.argwhere(binary > 0)
        if active_positions.size <= 0:
            return []
        pixel_ys = active_positions[:, 0].astype(np.float32)
        pixel_xs = active_positions[:, 1].astype(np.float32)
        local_xs = pixel_xs - float(min_x)
        local_ys = pixel_ys - float(min_y)
        quad_counts = [0, 0, 0, 0]
        radial_counts = [0, 0, 0, 0]
        edge_contacts = {"top": 0, "right": 0, "bottom": 0, "left": 0}
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        max_radius = math.sqrt(max(1.0, ((bbox_w - 1) * 0.5) ** 2 + ((bbox_h - 1) * 0.5) ** 2))
        quad_ids = ((local_ys >= bbox_h * 0.5).astype(np.int32) * 2 + (local_xs >= bbox_w * 0.5).astype(np.int32)).astype(np.int32)
        quad_hist = np.bincount(quad_ids, minlength=4)
        quad_counts = [int(value) for value in quad_hist[:4]]
        radii = np.sqrt((pixel_xs - center_x) ** 2 + (pixel_ys - center_y) ** 2) / max_radius
        radial_ids = np.clip(np.floor(radii * 4.0).astype(np.int32), 0, 3)
        radial_hist = np.bincount(radial_ids, minlength=4)
        radial_counts = [int(value) for value in radial_hist[:4]]
        edge_contacts["top"] = int(np.count_nonzero(pixel_ys <= (min_y + 1)))
        edge_contacts["bottom"] = int(np.count_nonzero(pixel_ys >= (max_y - 1)))
        edge_contacts["left"] = int(np.count_nonzero(pixel_xs <= (min_x + 1)))
        edge_contacts["right"] = int(np.count_nonzero(pixel_xs >= (max_x - 1)))

        coarse_grid = self._coarse_binary_grid_array(bbox_binary, target_cols=20, target_rows=20)
        hole_count = self._hole_count_from_grid(coarse_grid.tolist())

        def bin4(value: float) -> int:
            return int(max(0, min(3, math.floor(_clamp(value, 0.0, 0.9999) * 4.0))))

        quad_sig = "".join(str(bin4(count / max(1.0, float(active_count)))) for count in quad_counts)
        radial_sig = "".join(str(bin4(count / max(1.0, float(active_count)))) for count in radial_counts)
        edge_sig = "".join(
            str(
                bin4(
                    edge_contacts[key]
                    / max(1.0, float(bbox_w if key in {"top", "bottom"} else bbox_h))
                )
            )
            for key in ("top", "right", "bottom", "left")
        )
        shape_sig = (
            f"h{max(0, min(3, int(hole_count)))}"
            f"_c{bin4(center_void)}"
            f"_hs{bin4(horizontal_symmetry)}"
            f"_vs{bin4(vertical_symmetry)}"
            f"_d{bin4(ink_density * 4.0)}"
            f"_f{bin4(bbox_fill)}"
            f"_a{bin4(min(1.0, aspect / 2.0))}"
        )
        bbox_sig = (
            f"x{bin4(bbox_w / max(1.0, float(width)))}"
            f"_y{bin4(bbox_h / max(1.0, float(height)))}"
            f"_cx{bin4(bbox_cx)}"
            f"_cy{bin4(bbox_cy)}"
        )
        polarity_tag = "bright" if foreground_is_bright else "dark"
        global_confidence = _clamp(0.35 + min(0.42, ink_density * 2.6) + min(0.28, std_value / 255.0), 0.32, 1.0)
        coords = {
            "x": _round4(max(0.0, min(1.0, bbox_cx))),
            "y": _round4(max(0.0, min(1.0, bbox_cy))),
            "cx": _round4(max(0.0, min(1.0, bbox_cx))),
            "cy": _round4(max(0.0, min(1.0, bbox_cy))),
            "screen_x": _round4(min_x / max(1.0, float(width))),
            "screen_y": _round4(min_y / max(1.0, float(height))),
            "screen_w": _round4(bbox_w / max(1.0, float(width))),
            "screen_h": _round4(bbox_h / max(1.0, float(height))),
            "dx_from_gaze": _round4(bbox_cx - self.gaze_center[0]),
            "dy_from_gaze": _round4(bbox_cy - self.gaze_center[1]),
            "dr_from_gaze": _round4(math.sqrt((bbox_cx - self.gaze_center[0]) ** 2 + (bbox_cy - self.gaze_center[1]) ** 2)),
        }
        shared_attrs = {
            "sample_role": "global_structure",
            "foreground_polarity": polarity_tag,
            "global_confidence": _round4(global_confidence),
            "ink_density": _round4(ink_density),
            "bbox_fill": _round4(bbox_fill),
            "aspect_ratio": _round4(aspect),
            "center_void": _round4(center_void),
            "horizontal_symmetry": _round4(horizontal_symmetry),
            "vertical_symmetry": _round4(vertical_symmetry),
            "hole_count": int(hole_count),
            "proj_h_bin": proj_h,
            "proj_v_bin": proj_v,
            "quadrant_bin": quad_sig,
            "radial_bin": radial_sig,
            "edge_contact_bin": edge_sig,
            "bbox_signature": bbox_sig,
        }

        feature_specs = [
            ("shape", f"global_shape::{shape_sig}", 0.96),
            ("projection_h", f"global_proj_h::{proj_h}", 0.82),
            ("projection_v", f"global_proj_v::{proj_v}", 0.82),
            ("mass", f"global_mass::{quad_sig}::r{radial_sig}", 0.78),
            ("edge_contact", f"global_edge::{edge_sig}::{polarity_tag}::{bbox_sig}", 0.72),
        ]
        samples: list[dict[str, Any]] = []
        for index, (group, code, gain) in enumerate(feature_specs):
            attrs = dict(shared_attrs)
            attrs["global_feature_group"] = group
            attrs["global_feature_code"] = code
            attrs["sample_reason"] = "global_structure"
            samples.append(
                {
                    "sa_label": f"vision_mem::{code}",
                    "display_text": f"视觉全局特征[{code}]",
                    "energy": _round4(_clamp(global_confidence * gain, 0.18, 1.15)),
                    "position": int(width * height + index),
                    "source_type": source_type,
                    "sa_kind": "visual_global_feature_unit",
                    "channel": "vision",
                    "coords": dict(coords),
                    "attributes": attrs,
                }
            )
        return samples

    def _four_bin_signature(self, values: list[float]) -> str:
        if not values:
            return "0000"
        seg = max(1, int(math.ceil(len(values) / 4.0)))
        bins: list[str] = []
        for start in range(0, len(values), seg):
            part = values[start : start + seg]
            mean = sum(part) / max(1.0, float(len(part)))
            bins.append(str(int(max(0, min(3, math.floor(_clamp(mean, 0.0, 0.9999) * 4.0))))))
        return "".join((bins + ["0", "0", "0", "0"])[:4])

    def _four_bin_signature_array(self, values: np.ndarray) -> str:
        if values is None or values.size <= 0:
            return "0000"
        bins: list[str] = []
        for part in np.array_split(values.astype(np.float32, copy=False), 4):
            if part.size <= 0:
                bins.append("0")
                continue
            mean = float(np.mean(part))
            bins.append(str(int(max(0, min(3, math.floor(_clamp(mean, 0.0, 0.9999) * 4.0))))))
        return "".join((bins + ["0", "0", "0", "0"])[:4])

    def _axis_symmetry(
        self,
        binary_rows: list[list[int]],
        *,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
        axis: str,
    ) -> float:
        if not binary_rows:
            return 0.0
        diff = 0.0
        count = 0
        if axis == "horizontal":
            for offset in range(0, (max_y - min_y + 1) // 2):
                top = min_y + offset
                bottom = max_y - offset
                for pixel_x in range(min_x, max_x + 1):
                    diff += abs(int(binary_rows[top][pixel_x]) - int(binary_rows[bottom][pixel_x]))
                    count += 1
        else:
            for offset in range(0, (max_x - min_x + 1) // 2):
                left = min_x + offset
                right = max_x - offset
                for pixel_y in range(min_y, max_y + 1):
                    diff += abs(int(binary_rows[pixel_y][left]) - int(binary_rows[pixel_y][right]))
                    count += 1
        if count <= 0:
            return 0.0
        return _clamp(1.0 - diff / float(count), 0.0, 1.0)

    def _axis_symmetry_array(self, binary: np.ndarray, *, axis: str) -> float:
        if binary is None or binary.size <= 0 or binary.ndim != 2:
            return 0.0
        rows, cols = binary.shape
        if axis == "horizontal":
            if rows <= 1:
                return 0.0
            top = binary[: rows // 2, :]
            bottom = np.flipud(binary[-(rows // 2) :, :])
            if top.size <= 0 or bottom.size <= 0:
                return 0.0
            return _clamp(1.0 - float(np.mean(np.abs(top - bottom))), 0.0, 1.0)
        if cols <= 1:
            return 0.0
        left_half = binary[:, : cols // 2]
        right_half = np.fliplr(binary[:, -(cols // 2) :])
        if left_half.size <= 0 or right_half.size <= 0:
            return 0.0
        return _clamp(1.0 - float(np.mean(np.abs(left_half - right_half))), 0.0, 1.0)

    def _coarse_binary_grid(
        self,
        binary_rows: list[list[int]],
        *,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
        target_cols: int,
        target_rows: int,
    ) -> list[list[int]]:
        bbox_w = max(1, max_x - min_x + 1)
        bbox_h = max(1, max_y - min_y + 1)
        cols = max(4, min(int(target_cols), bbox_w))
        rows = max(4, min(int(target_rows), bbox_h))
        grid: list[list[int]] = []
        for row_index in range(rows):
            y0 = min_y + int(math.floor(row_index * bbox_h / float(rows)))
            y1 = min_y + int(math.floor((row_index + 1) * bbox_h / float(rows))) - 1
            if y1 < y0:
                y1 = y0
            row: list[int] = []
            for col_index in range(cols):
                x0 = min_x + int(math.floor(col_index * bbox_w / float(cols)))
                x1 = min_x + int(math.floor((col_index + 1) * bbox_w / float(cols))) - 1
                if x1 < x0:
                    x1 = x0
                hit = 0
                total = 0
                for pixel_y in range(y0, y1 + 1):
                    for pixel_x in range(x0, x1 + 1):
                        total += 1
                        hit += int(binary_rows[pixel_y][pixel_x])
                row.append(1 if hit / max(1.0, float(total)) >= 0.28 else 0)
            grid.append(row)
        return grid

    def _coarse_binary_grid_array(self, binary: np.ndarray, *, target_cols: int, target_rows: int) -> np.ndarray:
        if binary is None or binary.size <= 0 or binary.ndim != 2:
            return np.zeros((0, 0), dtype=np.uint8)
        bbox_h, bbox_w = binary.shape
        cols = max(4, min(int(target_cols), int(bbox_w)))
        rows = max(4, min(int(target_rows), int(bbox_h)))
        grid = np.zeros((rows, cols), dtype=np.uint8)
        for row_index in range(rows):
            y0 = int(math.floor(row_index * bbox_h / float(rows)))
            y1 = int(math.floor((row_index + 1) * bbox_h / float(rows)))
            if y1 <= y0:
                y1 = min(bbox_h, y0 + 1)
            for col_index in range(cols):
                x0 = int(math.floor(col_index * bbox_w / float(cols)))
                x1 = int(math.floor((col_index + 1) * bbox_w / float(cols)))
                if x1 <= x0:
                    x1 = min(bbox_w, x0 + 1)
                patch = binary[y0:y1, x0:x1]
                if patch.size > 0 and float(np.mean(patch)) >= 0.28:
                    grid[row_index, col_index] = 1
        return grid

    def _hole_count_from_grid(self, grid: list[list[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows = len(grid)
        cols = len(grid[0])
        visited: set[tuple[int, int]] = set()
        holes = 0
        for row_index in range(rows):
            for col_index in range(cols):
                if grid[row_index][col_index] != 0 or (row_index, col_index) in visited:
                    continue
                queue = [(row_index, col_index)]
                visited.add((row_index, col_index))
                touches_border = False
                while queue:
                    cur_row, cur_col = queue.pop()
                    if cur_row in {0, rows - 1} or cur_col in {0, cols - 1}:
                        touches_border = True
                    for next_row, next_col in (
                        (cur_row - 1, cur_col),
                        (cur_row + 1, cur_col),
                        (cur_row, cur_col - 1),
                        (cur_row, cur_col + 1),
                    ):
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if grid[next_row][next_col] != 0 or (next_row, next_col) in visited:
                            continue
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col))
                if not touches_border:
                    holes += 1
        return holes

    def _update_fixation_buffer(self, raw_samples: list[dict[str, Any]], *, tick_index: int) -> None:
        next_buffer: dict[tuple[int, int], dict[str, Any]] = {}
        for key, row in self._fixation_buffer.items():
            age = max(0, int(tick_index) - int(row.get("last_seen_tick", tick_index) or tick_index))
            energy = float(row.get("energy", 0.0) or 0.0) * (0.92 ** age)
            if energy < 0.02:
                continue
            next_row = dict(row)
            next_row["energy"] = _round4(energy)
            next_buffer[key] = next_row

        for item in raw_samples:
            coords = dict(item.get("coords", {}) or {})
            attrs = dict(item.get("attributes", {}) or {})
            key = (int(coords.get("pixel_x", -1) or -1), int(coords.get("pixel_y", -1) or -1))
            if key[0] < 0 or key[1] < 0:
                continue
            row = next_buffer.get(key, {})
            next_buffer[key] = {
                "energy": _round4(float(row.get("energy", 0.0) or 0.0) + float(item.get("energy", 0.0) or 0.0)),
                "brightness": _round4(float(attrs.get("brightness", 0.0) or 0.0)),
                "avg_r": _round4(float(attrs.get("avg_r", 0.0) or 0.0)),
                "avg_g": _round4(float(attrs.get("avg_g", 0.0) or 0.0)),
                "avg_b": _round4(float(attrs.get("avg_b", 0.0) or 0.0)),
                "last_seen_tick": int(tick_index),
                "sample_hits": int(row.get("sample_hits", 0) or 0) + 1,
                "source_tag": str(item.get("source_type", "") or ""),
                "sample_reason": str(attrs.get("sample_reason", "") or ""),
            }
        if len(next_buffer) > self.reconstruction_patch_budget * 6:
            rows = sorted(
                next_buffer.items(),
                key=lambda item: (-float((item[1] or {}).get("energy", 0.0) or 0.0), -int((item[1] or {}).get("last_seen_tick", -1) or -1)),
            )
            next_buffer = dict(rows[: self.reconstruction_patch_budget * 6])
        self._fixation_buffer = next_buffer

    def _export_fixation_cells(self, *, width: int, height: int) -> list[dict[str, Any]]:
        rows = sorted(
            self._fixation_buffer.items(),
            key=lambda item: (-float((item[1] or {}).get("energy", 0.0) or 0.0), -int((item[1] or {}).get("last_seen_tick", -1) or -1), item[0]),
        )
        cells: list[dict[str, Any]] = []
        for (pixel_x, pixel_y), row in rows[: self.reconstruction_patch_budget * 2]:
            screen_x = pixel_x / max(1.0, float(width))
            screen_y = pixel_y / max(1.0, float(height))
            screen_w = 1.0 / max(1.0, float(width))
            screen_h = 1.0 / max(1.0, float(height))
            cells.append(
                {
                    "row": int(pixel_y),
                    "col": int(pixel_x),
                    "pixel_x": int(pixel_x),
                    "pixel_y": int(pixel_y),
                    "screen_x": _round4(screen_x),
                    "screen_y": _round4(screen_y),
                    "screen_w": _round4(screen_w),
                    "screen_h": _round4(screen_h),
                    "cx": _round4(screen_x + screen_w * 0.5),
                    "cy": _round4(screen_y + screen_h * 0.5),
                    "avg_r": _round4(float(row.get("avg_r", 0.0) or 0.0)),
                    "avg_g": _round4(float(row.get("avg_g", 0.0) or 0.0)),
                    "avg_b": _round4(float(row.get("avg_b", 0.0) or 0.0)),
                    "brightness": _round4(float(row.get("brightness", 0.0) or 0.0)),
                    "motion": 0.0,
                    "energy": _round4(float(row.get("energy", 0.0) or 0.0)),
                    "sample_hits": int(row.get("sample_hits", 0) or 0),
                    "sample_reason": str(row.get("sample_reason", "") or ""),
                }
            )
        return cells

    def _push_recent_shape_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self._recent_shape_candidate_ring.append([_clone_sa_item(item) for item in candidates if isinstance(item, dict)])
        self._recent_shape_candidate_ring = self._recent_shape_candidate_ring[-self.dynamic_track_window :]

    def _estimate_global_motion(self, shape_candidates: list[dict[str, Any]]) -> dict[str, float]:
        previous_frame = self._recent_shape_candidate_ring[-1] if self._recent_shape_candidate_ring else []
        vectors: list[tuple[float, float, float]] = []
        if previous_frame and shape_candidates:
            used_prev_ids: set[str] = set()
            for current in shape_candidates:
                best_prev: dict[str, Any] | None = None
                best_score = -1.0
                current_coords = dict(current.get("coords", {}) or {})
                current_attrs = dict(current.get("attributes", {}) or {})
                current_cx = float(current_coords.get("cx", 0.5) or 0.5)
                current_cy = float(current_coords.get("cy", 0.5) or 0.5)
                current_area = float(current_attrs.get("area_ratio", 0.0) or 0.0)
                current_rgb = (
                    float(current_attrs.get("avg_r", 0.0) or 0.0),
                    float(current_attrs.get("avg_g", 0.0) or 0.0),
                    float(current_attrs.get("avg_b", 0.0) or 0.0),
                )
                for previous in previous_frame:
                    prev_id = str(previous.get("candidate_id", "") or "")
                    if prev_id and prev_id in used_prev_ids:
                        continue
                    prev_coords = dict(previous.get("coords", {}) or {})
                    prev_attrs = dict(previous.get("attributes", {}) or {})
                    prev_cx = float(prev_coords.get("cx", 0.5) or 0.5)
                    prev_cy = float(prev_coords.get("cy", 0.5) or 0.5)
                    displacement = math.sqrt((current_cx - prev_cx) ** 2 + (current_cy - prev_cy) ** 2)
                    pos_sim = max(0.0, 1.0 - displacement * 4.2)
                    shape_sim = self._shape_similarity(current, previous)
                    prev_area = float(prev_attrs.get("area_ratio", 0.0) or 0.0)
                    size_sim = 1.0 - min(1.0, abs(current_area - prev_area) / max(0.02, prev_area + 0.02))
                    prev_rgb = (
                        float(prev_attrs.get("avg_r", 0.0) or 0.0),
                        float(prev_attrs.get("avg_g", 0.0) or 0.0),
                        float(prev_attrs.get("avg_b", 0.0) or 0.0),
                    )
                    color_sim = max(0.0, 1.0 - _difference_score(current_rgb, prev_rgb) * 1.8)
                    score = shape_sim * 0.52 + pos_sim * 0.24 + size_sim * 0.14 + color_sim * 0.10
                    if score > best_score:
                        best_score = score
                        best_prev = previous
                if best_prev is None or best_score < 0.46:
                    continue
                best_prev_id = str(best_prev.get("candidate_id", "") or "")
                if best_prev_id:
                    used_prev_ids.add(best_prev_id)
                prev_coords = dict(best_prev.get("coords", {}) or {})
                prev_attrs = dict(best_prev.get("attributes", {}) or {})
                dx = current_cx - float(prev_coords.get("cx", current_cx) or current_cx)
                dy = current_cy - float(prev_coords.get("cy", current_cy) or current_cy)
                weight = max(
                    0.05,
                    best_score * 0.55
                    + float(current.get("energy", 0.0) or 0.0) * 0.15
                    + float(best_prev.get("energy", 0.0) or 0.0) * 0.10
                    + float(current_attrs.get("structure_discriminability", 0.0) or 0.0) * 0.10
                    + float(prev_attrs.get("structure_discriminability", 0.0) or 0.0) * 0.10,
                )
                vectors.append((dx, dy, weight))
        if not vectors:
            row = {"dx": 0.0, "dy": 0.0, "speed": 0.0}
            self._global_motion_history.append(row)
            self._global_motion_history = self._global_motion_history[-self.dynamic_track_window :]
            return row
        vectors.sort(key=lambda item: item[2], reverse=True)
        top = vectors[: min(24, len(vectors))]
        sorted_dx = sorted(item[0] for item in top)
        sorted_dy = sorted(item[1] for item in top)
        dx = sorted_dx[len(sorted_dx) // 2]
        dy = sorted_dy[len(sorted_dy) // 2]
        history_dx = [float(row.get("dx", 0.0) or 0.0) for row in self._global_motion_history[-2:]]
        history_dy = [float(row.get("dy", 0.0) or 0.0) for row in self._global_motion_history[-2:]]
        if history_dx:
            dx = dx * 0.76 + (sum(history_dx) / max(1.0, float(len(history_dx)))) * 0.24
        if history_dy:
            dy = dy * 0.76 + (sum(history_dy) / max(1.0, float(len(history_dy)))) * 0.24
        row = {"dx": _round4(dx), "dy": _round4(dy), "speed": _round4(math.sqrt(dx * dx + dy * dy))}
        self._global_motion_history.append(row)
        self._global_motion_history = self._global_motion_history[-self.dynamic_track_window :]
        return row

    def _build_shape_candidates(
        self,
        *,
        image: Image.Image,
        raw_samples: list[dict[str, Any]],
        global_structure_samples: list[dict[str, Any]],
        contour_bundle: dict[str, Any] | None,
        source_type: str,
    ) -> list[dict[str, Any]]:
        contour_bundle = dict(contour_bundle or {})
        contour_components = [dict(item) for item in (contour_bundle.get("components", []) or []) if isinstance(item, dict)]
        if contour_components:
            contour_candidates: list[dict[str, Any]] = []
            max_candidates = (
                self.dynamic_candidate_limit_focus
                if str(self._attention_mode or "background") == "visual_focus" or bool(self._attention_boost.get("active", False))
                else self.dynamic_candidate_limit_background
            )
            for index, component in enumerate(contour_components[:max_candidates]):
                coords = dict(component.get("coords", {}) or {})
                mean_rgb = dict(component.get("mean_rgb", {}) or {})
                frame_change = self._frame_change_score_for_coords(coords)
                component_motion = _clamp(frame_change, 0.0, 1.0)
                contour_candidates.append(
                    {
                        "candidate_id": f"shape_cand::contour::{self._sensor_tick}_{index}",
                        "sa_label": f"vision_dyn_shape::contour::{self._sensor_tick}_{index}",
                        "display_text": f"轮廓候选[{index + 1}]",
                        "energy": _round4(_clamp(0.24 + float(component.get("area_ratio", 0.0) or 0.0) * 3.2, 0.12, 1.25)),
                        "source_type": source_type,
                        "channel": "vision",
                        "sa_kind": "visual_dynamic_candidate_unit",
                        "coords": coords,
                        "attributes": {
                            "sample_role": "dynamic_shape_candidate",
                            "candidate_confidence": _round4(_clamp(0.44 + float(component.get("solidity", 0.0) or 0.0) * 0.30 + float(component.get("bbox_fill", 0.0) or 0.0) * 0.18, 0.0, 1.0)),
                            "cluster_sample_count": int(max(2, round(float(component.get("area_ratio", 0.0) or 0.0) * 300.0))),
                            "avg_r": _round4(float(mean_rgb.get("r", 0.0) or 0.0)),
                            "avg_g": _round4(float(mean_rgb.get("g", 0.0) or 0.0)),
                            "avg_b": _round4(float(mean_rgb.get("b", 0.0) or 0.0)),
                            "brightness": _round4(
                                0.299 * float(mean_rgb.get("r", 0.0) or 0.0)
                                + 0.587 * float(mean_rgb.get("g", 0.0) or 0.0)
                                + 0.114 * float(mean_rgb.get("b", 0.0) or 0.0)
                            ),
                            "edge_strength": _round4(min(1.0, 0.34 + float(component.get("solidity", 0.0) or 0.0) * 0.38)),
                            "stroke_likeness": _round4(min(1.0, 0.22 + float(component.get("extent", 0.0) or 0.0) * 0.44)),
                            "endpoint_likeness": _round4(max(0.0, 0.46 - float(component.get("roundness", 0.0) or 0.0) * 0.28)),
                            "corner_likeness": _round4(max(0.0, 0.38 + abs(float(component.get("aspect_ratio", 1.0) or 1.0) - 1.0) * 0.14)),
                            "opening_likeness": _round4(max(0.0, 0.42 - float(component.get("hole_like", 0.0) or 0.0) * 0.18)),
                            "closure_likeness": _round4(min(1.0, 0.36 + float(component.get("bbox_fill", 0.0) or 0.0) * 0.44)),
                            "arc_balance": _round4(min(1.0, 0.28 + float(component.get("roundness", 0.0) or 0.0) * 0.58)),
                            "structure_discriminability": _round4(
                                _clamp(
                                    0.36
                                    + float(component.get("hole_like", 0.0) or 0.0) * 0.22
                                    + abs(float(component.get("aspect_ratio", 1.0) or 1.0) - 1.0) * 0.16
                                    + float(component.get("solidity", 0.0) or 0.0) * 0.18,
                                    0.0,
                                    1.0,
                                )
                            ),
                            "straight_likeness": _round4(max(0.0, 0.48 - float(component.get("roundness", 0.0) or 0.0) * 0.26)),
                            "curvilinear_likeness": _round4(min(1.0, 0.24 + float(component.get("roundness", 0.0) or 0.0) * 0.62)),
                            "angularity": _round4(max(0.0, 0.44 - float(component.get("roundness", 0.0) or 0.0) * 0.24)),
                            "roundness": _round4(float(component.get("roundness", 0.0) or 0.0)),
                            "local_symmetry": _round4(max(float(component.get("horizontal_symmetry", 0.0) or 0.0), float(component.get("vertical_symmetry", 0.0) or 0.0))),
                            "horizontal_symmetry": _round4(float(component.get("horizontal_symmetry", 0.0) or 0.0)),
                            "vertical_symmetry": _round4(float(component.get("vertical_symmetry", 0.0) or 0.0)),
                            "opening_dir_x": 0.0,
                            "opening_dir_y": 0.0,
                            "opening_direction_strength": 0.0,
                            "proj_h_bin": str(component.get("proj_h_bin", "") or "0000"),
                            "proj_v_bin": str(component.get("proj_v_bin", "") or "0000"),
                            "radial_bin": str(component.get("radial_bin", "") or "0000"),
                            "quadrant_bin": str(component.get("quadrant_bin", "") or "0000"),
                            "foreground_polarity": str(component.get("foreground_polarity", "bright") or "bright"),
                            "orient_hist_bin": str(component.get("hu_signature", "") or "0000")[:4].ljust(4, "0"),
                            "radial_hist_bin": str(component.get("radial_signature", "") or "0000")[:4].ljust(4, "0"),
                            "local_patch_signature": str(component.get("hu_signature", "") or "0000000")[:9],
                        "area_ratio": _round4(float(component.get("area_ratio", 0.0) or 0.0)),
                        "bbox_fill": _round4(float(component.get("bbox_fill", 0.0) or 0.0)),
                        "aspect_ratio": _round4(float(component.get("aspect_ratio", 0.0) or 0.0)),
                        "hole_like": _round4(float(component.get("hole_like", 0.0) or 0.0)),
                        "center_void": _round4(float(component.get("center_void", 0.0) or 0.0)),
                        "hu_signature": str(component.get("hu_signature", "") or "0000000"),
                        "radial_signature": str(component.get("radial_signature", "") or "00000000"),
                        "edge_contact_bin": str(component.get("edge_contact_bin", "") or "0000"),
                        "bbox_signature": str(component.get("bbox_signature", "") or "x0_y0_w0_h0"),
                        "motion_strength": _round4(float(component.get("motion_strength", 0.0) or 0.0)),
                        "motion_peak": _round4(float(component.get("motion_peak", 0.0) or 0.0)),
                        "motion": _round4(component_motion),
                        "frame_change": _round4(frame_change),
                    },
                }
            )
            if contour_candidates:
                return contour_candidates
        if not raw_samples:
            return []
        max_candidates = (
            self.dynamic_candidate_limit_focus
            if str(self._attention_mode or "background") == "visual_focus" or bool(self._attention_boost.get("active", False))
            else self.dynamic_candidate_limit_background
        )
        cluster_radius = 0.065 if max_candidates > self.dynamic_candidate_limit_background else 0.08
        source_limit = min(len(raw_samples), max_candidates * 8)
        ranked = sorted(
            raw_samples,
            key=lambda item: (
                -(
                    float((item.get("attributes", {}) or {}).get("structure_priority", 0.0) or 0.0) * 0.40
                    + float((item.get("attributes", {}) or {}).get("edge_priority", 0.0) or 0.0) * 0.22
                    + float((item.get("attributes", {}) or {}).get("stroke_priority", 0.0) or 0.0) * 0.18
                    + float((item.get("attributes", {}) or {}).get("motion", 0.0) or 0.0) * 0.12
                    + float(item.get("energy", 0.0) or 0.0) * 0.08
                ),
                str(item.get("sa_label", "") or ""),
            ),
        )[:source_limit]
        ranked = [self._ensure_visual_descriptors(image=image, item=item) for item in ranked]
        global_shape = next(
            (
                item
                for item in global_structure_samples
                if str(((item.get("attributes", {}) or {}).get("global_feature_group", "") or "")) == "shape"
            ),
            None,
        )
        global_attrs = dict((global_shape or {}).get("attributes", {}) or {})
        numeric_keys = (
            "avg_r",
            "avg_g",
            "avg_b",
            "brightness",
            "edge_strength",
            "stroke_likeness",
            "endpoint_likeness",
            "corner_likeness",
            "opening_likeness",
            "closure_likeness",
            "arc_balance",
            "structure_discriminability",
            "straight_likeness",
            "curvilinear_likeness",
            "angularity",
            "roundness",
            "opening_dir_x",
            "opening_dir_y",
            "opening_direction_strength",
            "local_symmetry",
            "horizontal_symmetry",
            "vertical_symmetry",
            "hole_like",
            "center_void",
            "motion",
        )
        token_keys = (
            "proj_h_bin",
            "proj_v_bin",
            "radial_bin",
            "quadrant_bin",
            "foreground_polarity",
            "orient_hist_bin",
            "radial_hist_bin",
            "local_patch_signature",
        )
        clusters: list[dict[str, Any]] = []

        def _weighted_score(item: dict[str, Any]) -> float:
            attrs = dict(item.get("attributes", {}) or {})
            return max(
                0.05,
                float(attrs.get("structure_priority", 0.0) or 0.0) * 0.34
                + float(attrs.get("edge_priority", 0.0) or 0.0) * 0.22
                + float(attrs.get("stroke_priority", 0.0) or 0.0) * 0.16
                + float(attrs.get("structure_discriminability", 0.0) or 0.0) * 0.14
                + float(attrs.get("motion", 0.0) or 0.0) * 0.08
                + float(item.get("energy", 0.0) or 0.0) * 0.06,
            )

        def _cluster_center(cluster: dict[str, Any]) -> tuple[float, float]:
            weight_sum = max(1e-6, float(cluster.get("weight_sum", 0.0) or 0.0))
            return (
                float(cluster.get("cx_sum", 0.0) or 0.0) / weight_sum,
                float(cluster.get("cy_sum", 0.0) or 0.0) / weight_sum,
            )

        def _token_majority(cluster: dict[str, Any], key: str, fallback: str) -> str:
            counts = dict(((cluster.get("token_counts", {}) or {}).get(key, {}) or {}))
            if not counts:
                return fallback
            ordered = sorted(counts.items(), key=lambda row: (-int(row[1]), str(row[0])))
            return str(ordered[0][0] or fallback)

        def _new_cluster(item: dict[str, Any], weight: float) -> dict[str, Any]:
            coords = dict(item.get("coords", {}) or {})
            attrs = dict(item.get("attributes", {}) or {})
            token_counts = {
                key: {str(attrs.get(key, global_attrs.get(key, "")) or str(global_attrs.get(key, ""))): 1}
                for key in token_keys
            }
            return {
                "best_item": item,
                "best_weight": float(weight),
                "weight_sum": float(weight),
                "energy_sum": float(item.get("energy", 0.0) or 0.0) * float(weight),
                "sample_count": 1,
                "cx_sum": float(coords.get("cx", 0.5) or 0.5) * float(weight),
                "cy_sum": float(coords.get("cy", 0.5) or 0.5) * float(weight),
                "x_min": float(coords.get("screen_x", 0.0) or 0.0),
                "y_min": float(coords.get("screen_y", 0.0) or 0.0),
                "x_max": float(coords.get("screen_x", 0.0) or 0.0) + float(coords.get("screen_w", 0.0) or 0.0),
                "y_max": float(coords.get("screen_y", 0.0) or 0.0) + float(coords.get("screen_h", 0.0) or 0.0),
                "attr_sums": {key: float(attrs.get(key, 0.0) or 0.0) * float(weight) for key in numeric_keys},
                "token_counts": token_counts,
            }

        def _merge_into_cluster(cluster: dict[str, Any], item: dict[str, Any], weight: float) -> None:
            coords = dict(item.get("coords", {}) or {})
            attrs = dict(item.get("attributes", {}) or {})
            cluster["weight_sum"] = float(cluster.get("weight_sum", 0.0) or 0.0) + float(weight)
            cluster["energy_sum"] = float(cluster.get("energy_sum", 0.0) or 0.0) + float(item.get("energy", 0.0) or 0.0) * float(weight)
            cluster["sample_count"] = int(cluster.get("sample_count", 0) or 0) + 1
            cluster["cx_sum"] = float(cluster.get("cx_sum", 0.0) or 0.0) + float(coords.get("cx", 0.5) or 0.5) * float(weight)
            cluster["cy_sum"] = float(cluster.get("cy_sum", 0.0) or 0.0) + float(coords.get("cy", 0.5) or 0.5) * float(weight)
            cluster["x_min"] = min(float(cluster.get("x_min", 1.0) or 1.0), float(coords.get("screen_x", 0.0) or 0.0))
            cluster["y_min"] = min(float(cluster.get("y_min", 1.0) or 1.0), float(coords.get("screen_y", 0.0) or 0.0))
            cluster["x_max"] = max(
                float(cluster.get("x_max", 0.0) or 0.0),
                float(coords.get("screen_x", 0.0) or 0.0) + float(coords.get("screen_w", 0.0) or 0.0),
            )
            cluster["y_max"] = max(
                float(cluster.get("y_max", 0.0) or 0.0),
                float(coords.get("screen_y", 0.0) or 0.0) + float(coords.get("screen_h", 0.0) or 0.0),
            )
            attr_sums = dict(cluster.get("attr_sums", {}) or {})
            for key in numeric_keys:
                attr_sums[key] = float(attr_sums.get(key, 0.0) or 0.0) + float(attrs.get(key, 0.0) or 0.0) * float(weight)
            cluster["attr_sums"] = attr_sums
            token_counts = dict(cluster.get("token_counts", {}) or {})
            for key in token_keys:
                bucket = token_counts.setdefault(key, {})
                token = str(attrs.get(key, global_attrs.get(key, "")) or str(global_attrs.get(key, "")))
                bucket[token] = int(bucket.get(token, 0) or 0) + 1
            cluster["token_counts"] = token_counts
            if float(weight) > float(cluster.get("best_weight", 0.0) or 0.0):
                cluster["best_item"] = item
                cluster["best_weight"] = float(weight)

        for item in ranked:
            coords = dict(item.get("coords", {}) or {})
            cx = float(coords.get("cx", 0.5) or 0.5)
            cy = float(coords.get("cy", 0.5) or 0.5)
            weight = _weighted_score(item)
            best_cluster: dict[str, Any] | None = None
            best_score = -1.0
            for cluster in clusters:
                center_x, center_y = _cluster_center(cluster)
                dist = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                if dist > cluster_radius * 1.8:
                    continue
                shape_sim = self._shape_similarity(item, dict(cluster.get("best_item", {}) or {}))
                color_sim = max(
                    0.0,
                    1.0
                    - _difference_score(
                        (
                            float((item.get("attributes", {}) or {}).get("avg_r", 0.0) or 0.0),
                            float((item.get("attributes", {}) or {}).get("avg_g", 0.0) or 0.0),
                            float((item.get("attributes", {}) or {}).get("avg_b", 0.0) or 0.0),
                        ),
                        (
                            float(((cluster.get("best_item", {}) or {}).get("attributes", {}) or {}).get("avg_r", 0.0) or 0.0),
                            float(((cluster.get("best_item", {}) or {}).get("attributes", {}) or {}).get("avg_g", 0.0) or 0.0),
                            float(((cluster.get("best_item", {}) or {}).get("attributes", {}) or {}).get("avg_b", 0.0) or 0.0),
                        ),
                    )
                    * 1.5,
                )
                score = max(0.0, 1.0 - dist / max(0.0001, cluster_radius)) * 0.56 + shape_sim * 0.32 + color_sim * 0.12
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
            if best_cluster is None or best_score < 0.48:
                if len(clusters) < max_candidates * 2:
                    clusters.append(_new_cluster(item, weight))
                continue
            _merge_into_cluster(best_cluster, item, weight)

        candidates: list[dict[str, Any]] = []
        ordered_clusters = sorted(
            clusters,
            key=lambda cluster: (
                -float(cluster.get("weight_sum", 0.0) or 0.0),
                -int(cluster.get("sample_count", 0) or 0),
            ),
        )[:max_candidates]
        for cluster_index, cluster in enumerate(ordered_clusters):
            center_x, center_y = _cluster_center(cluster)
            weight_sum = max(1e-6, float(cluster.get("weight_sum", 0.0) or 0.0))
            sample_count = int(cluster.get("sample_count", 0) or 0)
            padding = min(0.03, 0.008 + sample_count * 0.0015)
            screen_x = max(0.0, float(cluster.get("x_min", 0.0) or 0.0) - padding)
            screen_y = max(0.0, float(cluster.get("y_min", 0.0) or 0.0) - padding)
            screen_w = min(0.34, max(0.03, float(cluster.get("x_max", 0.0) or 0.0) - float(cluster.get("x_min", 0.0) or 0.0) + padding * 2.0))
            screen_h = min(0.34, max(0.03, float(cluster.get("y_max", 0.0) or 0.0) - float(cluster.get("y_min", 0.0) or 0.0) + padding * 2.0))
            area_ratio = screen_w * screen_h
            attr_sums = dict(cluster.get("attr_sums", {}) or {})
            mean_attrs = {key: _round4(float(attr_sums.get(key, 0.0) or 0.0) / weight_sum) for key in numeric_keys}
            cluster_confidence = _clamp(float(cluster.get("weight_sum", 0.0) or 0.0) / max(1.0, float(sample_count)) * 0.62 + min(0.28, sample_count * 0.022), 0.0, 1.0)
            candidates.append(
                {
                    "candidate_id": f"shape_cand::{self._sensor_tick}_{cluster_index}",
                    "sa_label": f"vision_dyn_shape::{self._sensor_tick}_{cluster_index}",
                    "display_text": f"动态候选[{cluster_index + 1}]",
                    "energy": _round4(_clamp(float(cluster.get("energy_sum", 0.0) or 0.0) / weight_sum, 0.06, 1.25)),
                    "source_type": source_type,
                    "channel": "vision",
                    "sa_kind": "visual_dynamic_candidate_unit",
                    "coords": {
                        "cx": _round4(center_x),
                        "cy": _round4(center_y),
                        "screen_x": _round4(screen_x),
                        "screen_y": _round4(screen_y),
                        "screen_w": _round4(screen_w),
                        "screen_h": _round4(screen_h),
                        "dx_from_gaze": _round4(center_x - self.gaze_center[0]),
                        "dy_from_gaze": _round4(center_y - self.gaze_center[1]),
                        "dr_from_gaze": _round4(math.sqrt((center_x - self.gaze_center[0]) ** 2 + (center_y - self.gaze_center[1]) ** 2)),
                    },
                    "attributes": {
                        "sample_role": "dynamic_shape_candidate",
                        "avg_r": _round4(float(mean_attrs.get("avg_r", 0.0) or 0.0)),
                        "avg_g": _round4(float(mean_attrs.get("avg_g", 0.0) or 0.0)),
                        "avg_b": _round4(float(mean_attrs.get("avg_b", 0.0) or 0.0)),
                        "brightness": _round4(float(mean_attrs.get("brightness", 0.0) or 0.0)),
                        "edge_strength": _round4(float(mean_attrs.get("edge_strength", 0.0) or 0.0)),
                        "stroke_likeness": _round4(float(mean_attrs.get("stroke_likeness", 0.0) or 0.0)),
                        "endpoint_likeness": _round4(float(mean_attrs.get("endpoint_likeness", 0.0) or 0.0)),
                        "corner_likeness": _round4(float(mean_attrs.get("corner_likeness", 0.0) or 0.0)),
                        "opening_likeness": _round4(float(mean_attrs.get("opening_likeness", 0.0) or 0.0)),
                        "closure_likeness": _round4(float(mean_attrs.get("closure_likeness", 0.0) or 0.0)),
                        "arc_balance": _round4(float(mean_attrs.get("arc_balance", 0.0) or 0.0)),
                        "structure_discriminability": _round4(float(mean_attrs.get("structure_discriminability", 0.0) or 0.0)),
                        "straight_likeness": _round4(float(mean_attrs.get("straight_likeness", 0.0) or 0.0)),
                        "curvilinear_likeness": _round4(float(mean_attrs.get("curvilinear_likeness", 0.0) or 0.0)),
                        "angularity": _round4(float(mean_attrs.get("angularity", 0.0) or 0.0)),
                        "roundness": _round4(float(mean_attrs.get("roundness", 0.0) or 0.0)),
                        "opening_dir_x": _round4(float(mean_attrs.get("opening_dir_x", 0.0) or 0.0)),
                        "opening_dir_y": _round4(float(mean_attrs.get("opening_dir_y", 0.0) or 0.0)),
                        "opening_direction_strength": _round4(float(mean_attrs.get("opening_direction_strength", 0.0) or 0.0)),
                        "local_symmetry": _round4(float(mean_attrs.get("local_symmetry", 0.0) or 0.0)),
                        "proj_h_bin": _token_majority(cluster, "proj_h_bin", str(global_attrs.get("proj_h_bin", "0000") or "0000")),
                        "proj_v_bin": _token_majority(cluster, "proj_v_bin", str(global_attrs.get("proj_v_bin", "0000") or "0000")),
                        "radial_bin": _token_majority(cluster, "radial_bin", str(global_attrs.get("radial_bin", "0000") or "0000")),
                        "quadrant_bin": _token_majority(cluster, "quadrant_bin", str(global_attrs.get("quadrant_bin", "0000") or "0000")),
                        "foreground_polarity": _token_majority(cluster, "foreground_polarity", str(global_attrs.get("foreground_polarity", "bright") or "bright")),
                        "orient_hist_bin": _token_majority(cluster, "orient_hist_bin", "0000"),
                        "radial_hist_bin": _token_majority(cluster, "radial_hist_bin", "0000"),
                        "local_patch_signature": _token_majority(
                            cluster,
                            "local_patch_signature",
                            str(((cluster.get("best_item", {}) or {}).get("attributes", {}) or {}).get("local_patch_signature", "") or "")[:9],
                        ),
                        "area_ratio": _round4(area_ratio),
                        "bbox_fill": _round4(_clamp(sample_count / max(1.0, area_ratio * 2600.0), 0.0, 1.0)),
                        "aspect_ratio": _round4(screen_w / max(0.0001, screen_h)),
                        "hole_like": _round4(float(mean_attrs.get("hole_like", 0.0) or 0.0)),
                        "center_void": _round4(float(mean_attrs.get("center_void", global_attrs.get("center_void", 0.0)) or 0.0)),
                        "horizontal_symmetry": _round4(float(mean_attrs.get("horizontal_symmetry", global_attrs.get("horizontal_symmetry", 0.0)) or 0.0)),
                        "vertical_symmetry": _round4(float(mean_attrs.get("vertical_symmetry", global_attrs.get("vertical_symmetry", 0.0)) or 0.0)),
                        "motion": _round4(float(mean_attrs.get("motion", 0.0) or 0.0)),
                        "candidate_confidence": _round4(cluster_confidence),
                        "cluster_sample_count": int(sample_count),
                    },
                }
            )
        return candidates

    def _compute_dynamic_objectness(
        self,
        *,
        coherence: float,
        boundary: float,
        persistence: float,
        shape_stability: float,
        motion_signal: float,
    ) -> float:
        return _clamp(
            motion_signal * 0.46
            + coherence * 0.16
            + boundary * 0.14
            + shape_stability * 0.06
            + persistence * 0.04,
            0.0,
            1.0,
        )

    def _shape_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        la = dict(left.get("attributes", {}) or {})
        ra = dict(right.get("attributes", {}) or {})
        scores = []
        for key in (
            "edge_strength",
            "stroke_likeness",
            "endpoint_likeness",
            "corner_likeness",
            "opening_likeness",
            "closure_likeness",
            "arc_balance",
            "structure_discriminability",
            "straight_likeness",
            "curvilinear_likeness",
            "angularity",
            "roundness",
            "local_symmetry",
            "horizontal_symmetry",
            "vertical_symmetry",
            "hole_like",
            "center_void",
            "opening_dir_x",
            "opening_dir_y",
            "opening_direction_strength",
        ):
            lv = float(la.get(key, 0.0) or 0.0)
            rv = float(ra.get(key, 0.0) or 0.0)
            scores.append(max(0.0, 1.0 - abs(lv - rv)))
        for key in (
            "proj_h_bin",
            "proj_v_bin",
            "radial_bin",
            "quadrant_bin",
            "foreground_polarity",
            "orient_hist_bin",
            "radial_hist_bin",
        ):
            scores.append(1.0 if str(la.get(key, "")) == str(ra.get(key, "")) else 0.0)
        return _clamp(sum(scores) / max(1.0, float(len(scores))), 0.0, 1.0)

    def _new_dynamic_track(
        self,
        *,
        candidate: dict[str, Any],
        tick_index: int,
        global_motion: dict[str, float],
        source_type: str,
    ) -> dict[str, Any]:
        self._dynamic_track_serial += 1
        attrs = dict(candidate.get("attributes", {}) or {})
        coords = dict(candidate.get("coords", {}) or {})
        motion = max(0.0, float(attrs.get("motion", 0.0) or 0.0) - float(global_motion.get("speed", 0.0) or 0.0) * 0.6)
        coherence = _clamp(
            0.35
            + float(attrs.get("edge_strength", 0.0) or 0.0) * 0.20
            + float(attrs.get("stroke_likeness", 0.0) or 0.0) * 0.10
            + motion * 0.18,
            0.0,
            1.0,
        )
        boundary = _clamp(
            float(attrs.get("edge_strength", 0.0) or 0.0) * 0.42
            + float(attrs.get("structure_discriminability", 0.0) or 0.0) * 0.28
            + motion * 0.22,
            0.0,
            1.0,
        )
        persistence = 1.0 / max(2.0, float(self.dynamic_track_window))
        shape_stability = _clamp(
            float(attrs.get("local_symmetry", 0.0) or 0.0) * 0.24
            + float(attrs.get("roundness", 0.0) or 0.0) * 0.18
            + float(attrs.get("straight_likeness", 0.0) or 0.0) * 0.18
            + float(attrs.get("curvilinear_likeness", 0.0) or 0.0) * 0.18
            + 0.12,
            0.0,
            1.0,
        )
        motion_signal = _clamp(max(motion * 0.9, motion * 5.5), 0.0, 1.0)
        dyn = self._compute_dynamic_objectness(
            coherence=coherence,
            boundary=boundary,
            persistence=persistence,
            shape_stability=shape_stability,
            motion_signal=motion_signal,
        )
        return {
            "track_id": f"trk_{self._dynamic_track_serial:04d}",
            "source_type": source_type,
            "sa_kind": "visual_dynamic_track_unit",
            "channel": "vision",
            "display_text": f"动态轨迹[{self._dynamic_track_serial}]",
            "energy": _round4(float(candidate.get("energy", 0.0) or 0.0)),
            "coords": dict(coords),
            "attributes": dict(attrs),
            "last_seen_tick": int(tick_index),
            "age": 1,
            "miss_count": 0,
            "velocity_dx": 0.0,
            "velocity_dy": 0.0,
            "velocity_dir_x": 0.0,
            "velocity_dir_y": 0.0,
            "speed": _round4(motion),
            "accel_dx": 0.0,
            "accel_dy": 0.0,
            "size_growth": 0.0,
            "motion_coherence": _round4(coherence),
            "boundary_motion_contrast": _round4(boundary),
            "temporal_persistence": _round4(persistence),
            "shape_stability": _round4(shape_stability),
            "dynamic_objectness": _round4(dyn),
            "gaze_relation_score": _round4(max(0.0, 1.0 - float(coords.get("dr_from_gaze", 1.0) or 1.0))),
        }

    def _merge_candidate_into_track(
        self,
        *,
        existing_track: dict[str, Any],
        candidate: dict[str, Any],
        tick_index: int,
        global_motion: dict[str, float],
    ) -> dict[str, Any]:
        merged = dict(existing_track)
        prev_coords = dict(existing_track.get("coords", {}) or {})
        coords = dict(candidate.get("coords", {}) or {})
        prev_attrs = dict(existing_track.get("attributes", {}) or {})
        attrs = dict(candidate.get("attributes", {}) or {})
        prev_cx = float(prev_coords.get("cx", 0.5) or 0.5)
        prev_cy = float(prev_coords.get("cy", 0.5) or 0.5)
        cx = float(coords.get("cx", prev_cx) or prev_cx)
        cy = float(coords.get("cy", prev_cy) or prev_cy)
        prev_dx = float(merged.get("velocity_dx", 0.0) or 0.0)
        prev_dy = float(merged.get("velocity_dy", 0.0) or 0.0)
        local_dx = cx - prev_cx - float(global_motion.get("dx", 0.0) or 0.0)
        local_dy = cy - prev_cy - float(global_motion.get("dy", 0.0) or 0.0)
        vel_dx = prev_dx * 0.58 + local_dx * 0.42
        vel_dy = prev_dy * 0.58 + local_dy * 0.42
        speed = math.sqrt(max(0.0, vel_dx * vel_dx + vel_dy * vel_dy))
        dir_norm = math.sqrt(max(1e-6, vel_dx * vel_dx + vel_dy * vel_dy))
        coherence = _clamp(
            float(existing_track.get("motion_coherence", 0.0) or 0.0) * 0.58
            + (
                0.32
                + float(attrs.get("edge_strength", 0.0) or 0.0) * 0.22
                + float(attrs.get("stroke_likeness", 0.0) or 0.0) * 0.12
                + max(0.0, float(attrs.get("motion", 0.0) or 0.0) - float(global_motion.get("speed", 0.0) or 0.0) * 0.6) * 0.18
            ) * 0.42,
            0.0,
            1.0,
        )
        boundary = _clamp(
            float(existing_track.get("boundary_motion_contrast", 0.0) or 0.0) * 0.60
            + (
                float(attrs.get("edge_strength", 0.0) or 0.0) * 0.42
                + float(attrs.get("structure_discriminability", 0.0) or 0.0) * 0.28
                + max(0.0, speed - float(global_motion.get("speed", 0.0) or 0.0)) * 0.26
            ) * 0.40,
            0.0,
            1.0,
        )
        shape_stability = _clamp(
            float(existing_track.get("shape_stability", 0.0) or 0.0) * 0.62
            + self._shape_similarity(candidate, existing_track) * 0.38,
            0.0,
            1.0,
        )
        persistence = _clamp(float(existing_track.get("temporal_persistence", 0.0) or 0.0) + 1.0 / max(2.0, float(self.dynamic_track_window)), 0.0, 1.0)
        local_motion = max(
            0.0,
            float(attrs.get("motion", 0.0) or 0.0) - float(global_motion.get("speed", 0.0) or 0.0) * 0.65,
        )
        motion_signal = _clamp(max(local_motion, speed * 5.8), 0.0, 1.0)
        dynamic_objectness = self._compute_dynamic_objectness(
            coherence=coherence,
            boundary=boundary,
            persistence=persistence,
            shape_stability=shape_stability,
            motion_signal=motion_signal,
        )
        merged["coords"] = {
            **prev_coords,
            **coords,
            "cx": _round4(prev_cx * 0.46 + cx * 0.54),
            "cy": _round4(prev_cy * 0.46 + cy * 0.54),
            "screen_x": _round4(float(coords.get("screen_x", prev_coords.get("screen_x", 0.0)) or prev_coords.get("screen_x", 0.0) or 0.0)),
            "screen_y": _round4(float(coords.get("screen_y", prev_coords.get("screen_y", 0.0)) or prev_coords.get("screen_y", 0.0) or 0.0)),
            "screen_w": _round4(float(coords.get("screen_w", prev_coords.get("screen_w", 0.05)) or prev_coords.get("screen_w", 0.05) or 0.05)),
            "screen_h": _round4(float(coords.get("screen_h", prev_coords.get("screen_h", 0.05)) or prev_coords.get("screen_h", 0.05) or 0.05)),
        }
        merged["attributes"] = {**prev_attrs, **attrs}
        merged["energy"] = _round4(float(existing_track.get("energy", 0.0) or 0.0) * 0.52 + float(candidate.get("energy", 0.0) or 0.0) * 0.48)
        merged["last_seen_tick"] = int(tick_index)
        merged["age"] = int(existing_track.get("age", 0) or 0) + 1
        merged["miss_count"] = 0
        merged["velocity_dx"] = _round4(vel_dx)
        merged["velocity_dy"] = _round4(vel_dy)
        merged["velocity_dir_x"] = _round4(vel_dx / dir_norm) if speed > 1e-6 else 0.0
        merged["velocity_dir_y"] = _round4(vel_dy / dir_norm) if speed > 1e-6 else 0.0
        merged["speed"] = _round4(speed)
        merged["accel_dx"] = _round4(vel_dx - prev_dx)
        merged["accel_dy"] = _round4(vel_dy - prev_dy)
        merged["size_growth"] = _round4(float(attrs.get("area_ratio", 0.0) or 0.0) - float(prev_attrs.get("area_ratio", 0.0) or 0.0))
        merged["motion_coherence"] = _round4(coherence)
        merged["boundary_motion_contrast"] = _round4(boundary)
        merged["temporal_persistence"] = _round4(persistence)
        merged["shape_stability"] = _round4(shape_stability)
        merged["dynamic_objectness"] = _round4(dynamic_objectness)
        merged["gaze_relation_score"] = _round4(max(0.0, 1.0 - float(merged["coords"].get("dr_from_gaze", 1.0) or 1.0)))
        return merged

    def _update_dynamic_tracks(
        self,
        *,
        tick_index: int,
        shape_candidates: list[dict[str, Any]],
        global_motion: dict[str, float],
        source_type: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        matched_tracks: set[str] = set()
        updated_tracks: dict[str, dict[str, Any]] = {}
        tracks = list(self._dynamic_shape_tracks.values())
        for candidate in shape_candidates:
            best_track_id = ""
            best_score = -1.0
            for track in tracks:
                track_id = str(track.get("track_id", "") or "")
                if not track_id or track_id in matched_tracks:
                    continue
                coords = dict(candidate.get("coords", {}) or {})
                tcoords = dict(track.get("coords", {}) or {})
                cx = float(coords.get("cx", 0.5) or 0.5)
                cy = float(coords.get("cy", 0.5) or 0.5)
                tx = float(tcoords.get("cx", 0.5) or 0.5)
                ty = float(tcoords.get("cy", 0.5) or 0.5)
                expected_x = tx + float(global_motion.get("dx", 0.0) or 0.0)
                expected_y = ty + float(global_motion.get("dy", 0.0) or 0.0)
                pos_sim = max(0.0, 1.0 - math.sqrt((cx - expected_x) ** 2 + (cy - expected_y) ** 2) * 5.0)
                shape_sim = self._shape_similarity(candidate, track)
                cand_attrs = dict(candidate.get("attributes", {}) or {})
                track_attrs = dict(track.get("attributes", {}) or {})
                cand_area = float(cand_attrs.get("area_ratio", 0.0) or 0.0)
                track_area = float(track_attrs.get("area_ratio", 0.0) or 0.0)
                size_sim = 1.0 - min(1.0, abs(cand_area - track_area) / max(0.02, track_area + 0.02))
                motion_sim = 1.0 - min(1.0, abs(float(cand_attrs.get("motion", 0.0) or 0.0) - float(track_attrs.get("motion", 0.0) or 0.0)))
                color_sim = max(
                    0.0,
                    1.0
                    - _difference_score(
                        (
                            float(cand_attrs.get("avg_r", 0.0) or 0.0),
                            float(cand_attrs.get("avg_g", 0.0) or 0.0),
                            float(cand_attrs.get("avg_b", 0.0) or 0.0),
                        ),
                        (
                            float(track_attrs.get("avg_r", 0.0) or 0.0),
                            float(track_attrs.get("avg_g", 0.0) or 0.0),
                            float(track_attrs.get("avg_b", 0.0) or 0.0),
                        ),
                    )
                    * 1.5,
                )
                gaze_sim = 1.0 - min(
                    1.0,
                    abs(float(coords.get("dr_from_gaze", 0.0) or 0.0) - float(tcoords.get("dr_from_gaze", 0.0) or 0.0)) * 1.8,
                )
                score = shape_sim * 0.40 + pos_sim * 0.28 + size_sim * 0.10 + color_sim * 0.08 + motion_sim * 0.08 + gaze_sim * 0.06
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
            if best_track_id and best_score >= self.dynamic_match_threshold:
                matched_tracks.add(best_track_id)
                existing = next((row for row in tracks if str(row.get("track_id", "") or "") == best_track_id), {})
                updated_tracks[best_track_id] = self._merge_candidate_into_track(
                    existing_track=existing,
                    candidate=candidate,
                    tick_index=tick_index,
                    global_motion=global_motion,
                )
            else:
                new_track = self._new_dynamic_track(
                    candidate=candidate,
                    tick_index=tick_index,
                    global_motion=global_motion,
                    source_type=source_type,
                )
                updated_tracks[str(new_track.get("track_id", "") or "")] = new_track
        for track in tracks:
            track_id = str(track.get("track_id", "") or "")
            if not track_id or track_id in updated_tracks:
                continue
            missed = _clone_sa_item(track)
            missed["miss_count"] = int(missed.get("miss_count", 0) or 0) + 1
            missed["age"] = int(missed.get("age", 0) or 0) + 1
            missed["dynamic_objectness"] = _round4(float(missed.get("dynamic_objectness", 0.0) or 0.0) * 0.82)
            missed["temporal_persistence"] = _round4(float(missed.get("temporal_persistence", 0.0) or 0.0) * 0.88)
            if int(missed.get("miss_count", 0) or 0) <= self.dynamic_track_forget_ticks and float(missed.get("dynamic_objectness", 0.0) or 0.0) >= 0.12:
                updated_tracks[track_id] = missed
        pruned = sorted(
            updated_tracks.values(),
            key=lambda row: (
                -float(row.get("dynamic_objectness", 0.0) or 0.0),
                -float(row.get("energy", 0.0) or 0.0),
                str(row.get("track_id", "") or ""),
            ),
        )[: self.dynamic_track_limit]
        self._dynamic_shape_tracks = {str(row.get("track_id", "") or ""): row for row in pruned if str(row.get("track_id", "") or "")}
        dynamic_tracks = [_clone_sa_item(row) for row in pruned]
        dynamic_tracks.sort(key=lambda row: (-float(row.get("dynamic_objectness", 0.0) or 0.0), str(row.get("track_id", "") or "")))
        dynamic_motion_samples = []
        for track in dynamic_tracks[: self.dynamic_summary_limit]:
            attrs = dict(track.get("attributes", {}) or {})
            coords = dict(track.get("coords", {}) or {})
            dyn = float(track.get("dynamic_objectness", 0.0) or 0.0)
            persistence = float(track.get("temporal_persistence", 0.0) or 0.0)
            coherence = float(track.get("motion_coherence", 0.0) or 0.0)
            boundary = float(track.get("boundary_motion_contrast", 0.0) or 0.0)
            speed = float(track.get("speed", 0.0) or 0.0)
            dynamic_motion_samples.append(
                {
                    "sa_label": f"vision_dyn::{track.get('track_id', '')}",
                    "display_text": f"动态对象[{track.get('track_id', '')}]",
                    "energy": _round4(_clamp(0.18 + dyn * 0.92, 0.12, 1.25)),
                    "source_type": source_type,
                    "channel": "vision",
                    "sa_kind": "visual_dynamic_track_unit",
                    "coords": dict(coords),
                    "attributes": {
                        "sample_role": "dynamic_motion_summary",
                        "track_id": str(track.get("track_id", "") or ""),
                        "motion_dx": _round4(float(track.get("velocity_dx", 0.0) or 0.0)),
                        "motion_dy": _round4(float(track.get("velocity_dy", 0.0) or 0.0)),
                        "motion_speed": _round4(speed),
                        "motion_dir_x": _round4(float(track.get("velocity_dir_x", 0.0) or 0.0)),
                        "motion_dir_y": _round4(float(track.get("velocity_dir_y", 0.0) or 0.0)),
                        "dynamic_objectness": _round4(dyn),
                        "temporal_persistence": _round4(persistence),
                        "motion_coherence": _round4(coherence),
                        "boundary_motion_contrast": _round4(boundary),
                        "shape_stability": _round4(float(track.get("shape_stability", 0.0) or 0.0)),
                        "gaze_relation_score": _round4(float(track.get("gaze_relation_score", 0.0) or 0.0)),
                        "motion_surprise": _round4(max(0.0, speed - persistence * 0.32)),
                        "edge_strength": _round4(float(attrs.get("edge_strength", 0.0) or 0.0)),
                        "stroke_likeness": _round4(float(attrs.get("stroke_likeness", 0.0) or 0.0)),
                        "endpoint_likeness": _round4(float(attrs.get("endpoint_likeness", 0.0) or 0.0)),
                        "corner_likeness": _round4(float(attrs.get("corner_likeness", 0.0) or 0.0)),
                        "opening_likeness": _round4(float(attrs.get("opening_likeness", 0.0) or 0.0)),
                        "closure_likeness": _round4(float(attrs.get("closure_likeness", 0.0) or 0.0)),
                        "arc_balance": _round4(float(attrs.get("arc_balance", 0.0) or 0.0)),
                        "structure_discriminability": _round4(float(attrs.get("structure_discriminability", 0.0) or 0.0)),
                        "straight_likeness": _round4(float(attrs.get("straight_likeness", 0.0) or 0.0)),
                        "curvilinear_likeness": _round4(float(attrs.get("curvilinear_likeness", 0.0) or 0.0)),
                        "angularity": _round4(float(attrs.get("angularity", 0.0) or 0.0)),
                        "roundness": _round4(float(attrs.get("roundness", 0.0) or 0.0)),
                        "local_symmetry": _round4(float(attrs.get("local_symmetry", 0.0) or 0.0)),
                        "horizontal_symmetry": _round4(float(attrs.get("horizontal_symmetry", 0.0) or 0.0)),
                        "vertical_symmetry": _round4(float(attrs.get("vertical_symmetry", 0.0) or 0.0)),
                        "opening_dir_x": _round4(float(attrs.get("opening_dir_x", 0.0) or 0.0)),
                        "opening_dir_y": _round4(float(attrs.get("opening_dir_y", 0.0) or 0.0)),
                        "opening_direction_strength": _round4(float(attrs.get("opening_direction_strength", 0.0) or 0.0)),
                        "hole_like": _round4(float(attrs.get("hole_like", 0.0) or 0.0)),
                        "center_void": _round4(float(attrs.get("center_void", 0.0) or 0.0)),
                        "proj_h_bin": str(attrs.get("proj_h_bin", "") or "0000")[:4],
                        "proj_v_bin": str(attrs.get("proj_v_bin", "") or "0000")[:4],
                        "orient_hist_bin": str(attrs.get("orient_hist_bin", "") or "0000")[:4],
                        "radial_hist_bin": str(attrs.get("radial_hist_bin", "") or "0000")[:4],
                        "radial_bin": str(attrs.get("radial_bin", "") or "0000")[:4],
                        "quadrant_bin": str(attrs.get("quadrant_bin", "") or "0000")[:4],
                        "foreground_polarity": str(attrs.get("foreground_polarity", "bright") or "bright"),
                        "local_patch_signature": str(attrs.get("local_patch_signature", "") or "")[:9],
                        "bbox_fill": _round4(float(attrs.get("bbox_fill", 0.0) or 0.0)),
                        "aspect_ratio": _round4(float(attrs.get("aspect_ratio", 0.0) or 0.0)),
                        "area_ratio": _round4(float(attrs.get("area_ratio", 0.0) or 0.0)),
                        "avg_r": _round4(float(attrs.get("avg_r", 0.0) or 0.0)),
                        "avg_g": _round4(float(attrs.get("avg_g", 0.0) or 0.0)),
                        "avg_b": _round4(float(attrs.get("avg_b", 0.0) or 0.0)),
                        "hu_signature": str(attrs.get("hu_signature", "") or "0000000"),
                        "radial_signature": str(attrs.get("radial_signature", "") or "00000000"),
                        "edge_contact_bin": str(attrs.get("edge_contact_bin", "") or "0000"),
                        "bbox_signature": str(attrs.get("bbox_signature", "") or "x0_y0_w0_h0"),
                        "motion_strength": _round4(float(attrs.get("motion_strength", 0.0) or 0.0)),
                        "motion_peak": _round4(float(attrs.get("motion_peak", 0.0) or 0.0)),
                    },
                }
            )
        object_count = sum(
            1
            for row in dynamic_tracks
            if float(row.get("dynamic_objectness", 0.0) or 0.0) >= 0.42
            and float(row.get("motion_coherence", 0.0) or 0.0) >= 0.40
            and float(row.get("boundary_motion_contrast", 0.0) or 0.0) >= 0.12
        )
        salience_mean = sum(float(row.get("dynamic_objectness", 0.0) or 0.0) for row in dynamic_tracks) / max(1.0, float(len(dynamic_tracks)))
        summary = {
            "track_count": len(dynamic_tracks),
            "object_count": object_count,
            "global_motion_dx": _round4(float(global_motion.get("dx", 0.0) or 0.0)),
            "global_motion_dy": _round4(float(global_motion.get("dy", 0.0) or 0.0)),
            "global_motion_speed": _round4(float(global_motion.get("speed", 0.0) or 0.0)),
            "dynamic_salience_mean": _round4(salience_mean),
            "preview": [str(item.get("sa_label", "") or "") for item in dynamic_motion_samples[:6]],
        }
        return dynamic_tracks, dynamic_motion_samples, summary
