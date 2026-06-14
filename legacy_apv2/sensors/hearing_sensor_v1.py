# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import audioop
import hashlib
import math
import wave
from collections import deque
from io import BytesIO
from typing import Any

import numpy as np


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_log2(value: float) -> float:
    return math.log(max(1e-6, float(value)), 2.0)


def _bin_code(value: float, bucket_count: int) -> int:
    count = max(1, int(bucket_count))
    scaled = int(math.floor(_clamp(float(value), 0.0, 1.0) * count))
    return max(0, min(count - 1, scaled))


class HearingSensorV1:
    def __init__(
        self,
        *,
        window_budget: int,
        window_ms: int = 50,
        focus_band_count: int = 12,
        focus_bandwidth_octaves: float = 1.15,
        attention_boost_enabled: bool = False,
        attention_boost_decay: float = 0.72,
        attention_boost_max_extra_window_budget: int = 0,
        attention_boost_max_extra_focus_budget: int = 0,
        attention_boost_min_bandwidth_scale: float = 0.35,
        attention_boost_focus_gain: float = 1.0,
        static_dedup_delta_threshold: float = 0.035,
        static_dedup_band_similarity_threshold: float = 0.08,
        static_dedup_max_suppression: float = 0.92,
        auditory_fatigue_decay: float = 0.82,
        auditory_fatigue_step: float = 0.12,
        auditory_fatigue_max: float = 1.0,
    ) -> None:
        self.window_budget = max(4, int(window_budget))
        self.window_ms = max(5, int(window_ms))
        self.focus_band_count = max(3, int(focus_band_count))
        self.focus_bandwidth_octaves = max(0.08, float(focus_bandwidth_octaves))
        self.attention_boost_enabled = bool(attention_boost_enabled)
        self.attention_boost_decay = _clamp(float(attention_boost_decay), 0.0, 1.0)
        self.attention_boost_max_extra_window_budget = max(0, int(attention_boost_max_extra_window_budget))
        self.attention_boost_max_extra_focus_budget = max(0, int(attention_boost_max_extra_focus_budget))
        self.attention_boost_min_bandwidth_scale = _clamp(float(attention_boost_min_bandwidth_scale), 0.05, 1.0)
        self.attention_boost_focus_gain = max(0.0, float(attention_boost_focus_gain))
        self.static_dedup_delta_threshold = max(0.0, float(static_dedup_delta_threshold))
        self.static_dedup_band_similarity_threshold = max(0.0, float(static_dedup_band_similarity_threshold))
        self.static_dedup_max_suppression = _clamp(float(static_dedup_max_suppression), 0.0, 0.999)
        self.auditory_fatigue_decay = _clamp(float(auditory_fatigue_decay), 0.0, 1.0)
        self.auditory_fatigue_step = max(0.0, float(auditory_fatigue_step))
        self.auditory_fatigue_max = max(0.0, float(auditory_fatigue_max))

        self._sensor_tick = 0
        self._stream_chunk_index = -1
        self._carry_frames: bytes = b""
        self._prev_window_state: dict[str, dict[str, float]] = {}
        self._recent_selected_windows: dict[str, int] = {}
        self._energy_history: deque[float] = deque(maxlen=24)
        self._audio_input_cache: dict[str, dict[str, Any]] = {}
        self._audio_input_cache_limit = 6
        self._audio_stream_cache: dict[str, dict[str, Any]] = {}
        self._audio_stream_cache_limit = 8
        self._audio_focus = {
            "center_hz": 1200.0,
            "bandwidth_octaves": self.focus_bandwidth_octaves,
        }
        self._attention_boost: dict[str, Any] = {
            "active": False,
            "strength": 0.0,
            "ticks_left": 0,
            "target_center_hz": _round4(self._audio_focus["center_hz"]),
            "target_bandwidth_octaves": _round4(self._audio_focus["bandwidth_octaves"]),
            "source_action": "",
            "window_budget_bonus": 0,
            "focus_budget_bonus": 0,
            "bandwidth_scale": 1.0,
            "focus_gain": 1.0,
        }
        self._attention_mode = "background"

    def set_attention_mode(self, mode: str) -> None:
        clean = str(mode or "").strip().lower()
        if clean not in {"background", "suppressed", "auditory_focus"}:
            clean = "background"
        self._attention_mode = clean

    def move_audio_focus(self, center_hz: float, *, bandwidth_octaves: float | None = None) -> None:
        center = _clamp(float(center_hz), 40.0, 12000.0)
        bandwidth = self.focus_bandwidth_octaves if bandwidth_octaves is None else max(0.08, float(bandwidth_octaves))
        self._audio_focus = {
            "center_hz": _round4(center),
            "bandwidth_octaves": _round4(bandwidth),
        }

    def audio_focus_snapshot(self) -> dict[str, Any]:
        return {
            "center_hz": _round4(float(self._audio_focus.get("center_hz", 1200.0) or 1200.0)),
            "bandwidth_octaves": _round4(float(self._audio_focus.get("bandwidth_octaves", self.focus_bandwidth_octaves) or self.focus_bandwidth_octaves)),
        }

    def apply_attention_boost(
        self,
        *,
        source_action: str,
        firmness_norm: float,
        target_center_hz: float | None = None,
        target_bandwidth_octaves: float | None = None,
    ) -> dict[str, Any]:
        if not self.attention_boost_enabled:
            return self.attention_boost_snapshot()
        strength = _clamp(float(firmness_norm), 0.0, 1.5)
        if strength <= 0.0:
            return self.attention_boost_snapshot()
        strength_norm = _clamp(strength / 1.0, 0.0, 1.0)
        center_hz = float(target_center_hz if target_center_hz is not None else self._audio_focus["center_hz"])
        bandwidth = float(target_bandwidth_octaves if target_bandwidth_octaves is not None else self._audio_focus["bandwidth_octaves"])
        window_bonus = int(round(self.attention_boost_max_extra_window_budget * strength_norm))
        focus_bonus = int(round(self.attention_boost_max_extra_focus_budget * strength_norm))
        bandwidth_scale = max(
            self.attention_boost_min_bandwidth_scale,
            1.0 - (1.0 - self.attention_boost_min_bandwidth_scale) * strength_norm,
        )
        focus_gain = 1.0 + max(0.0, self.attention_boost_focus_gain - 1.0) * strength_norm
        ticks_left = max(1, int(round(1 + strength_norm * 2.0)))
        self._attention_boost = {
            "active": True,
            "strength": _round4(strength_norm),
            "ticks_left": ticks_left,
            "target_center_hz": _round4(_clamp(center_hz, 40.0, 12000.0)),
            "target_bandwidth_octaves": _round4(max(0.08, bandwidth)),
            "source_action": str(source_action or ""),
            "window_budget_bonus": int(window_bonus),
            "focus_budget_bonus": int(focus_bonus),
            "bandwidth_scale": _round4(bandwidth_scale),
            "focus_gain": _round4(focus_gain),
        }
        return self.attention_boost_snapshot()

    def attention_boost_snapshot(self) -> dict[str, Any]:
        return {
            "active": bool(self._attention_boost.get("active", False)),
            "strength": _round4(float(self._attention_boost.get("strength", 0.0) or 0.0)),
            "ticks_left": int(self._attention_boost.get("ticks_left", 0) or 0),
            "target_center_hz": _round4(float(self._attention_boost.get("target_center_hz", self._audio_focus["center_hz"]) or self._audio_focus["center_hz"])),
            "target_bandwidth_octaves": _round4(float(self._attention_boost.get("target_bandwidth_octaves", self._audio_focus["bandwidth_octaves"]) or self._audio_focus["bandwidth_octaves"])),
            "source_action": str(self._attention_boost.get("source_action", "") or ""),
            "window_budget_bonus": int(self._attention_boost.get("window_budget_bonus", 0) or 0),
            "focus_budget_bonus": int(self._attention_boost.get("focus_budget_bonus", 0) or 0),
            "bandwidth_scale": _round4(float(self._attention_boost.get("bandwidth_scale", 1.0) or 1.0)),
            "focus_gain": _round4(float(self._attention_boost.get("focus_gain", 1.0) or 1.0)),
            "attention_mode": str(self._attention_mode or "background"),
        }

    def _effective_sampling_profile(self) -> dict[str, Any]:
        boost = self.attention_boost_snapshot()
        base_window_budget = max(4, min(int(self.window_budget), 12))
        if not bool(boost.get("active", False)):
            if str(self._attention_mode or "background") == "suppressed":
                return {
                    "window_budget": max(4, min(base_window_budget, 8)),
                    "focus_band_count": max(2, min(self.focus_band_count, 6)),
                    "bandwidth_octaves": max(0.12, float(self._audio_focus["bandwidth_octaves"])),
                    "focus_gain": 1.0,
                    "attention_mode": "suppressed",
                    "boost": boost,
                }
            return {
                "window_budget": int(base_window_budget),
                "focus_band_count": int(self.focus_band_count),
                "bandwidth_octaves": max(0.12, float(self._audio_focus["bandwidth_octaves"])),
                "focus_gain": 1.0,
                "attention_mode": "background",
                "boost": boost,
            }
        requested_budget = base_window_budget + int(boost.get("window_budget_bonus", 0) or 0)
        budget_cap = min(256, max(base_window_budget * 4, self.window_budget))
        return {
            "window_budget": max(4, min(requested_budget, budget_cap)),
            "focus_band_count": max(2, min(self.focus_band_count + int(boost.get("focus_budget_bonus", 0) or 0), 64)),
            "bandwidth_octaves": max(
                0.08,
                float(boost.get("target_bandwidth_octaves", self._audio_focus["bandwidth_octaves"]) or self._audio_focus["bandwidth_octaves"])
                * max(0.05, float(boost.get("bandwidth_scale", 1.0) or 1.0)),
            ),
            "focus_gain": max(1.0, float(boost.get("focus_gain", 1.0) or 1.0)),
            "attention_mode": "auditory_focus",
            "boost": boost,
        }

    def _decay_attention_boost(self) -> None:
        if not bool(self._attention_boost.get("active", False)):
            return
        ticks_left = max(0, int(self._attention_boost.get("ticks_left", 0) or 0) - 1)
        strength = max(0.0, float(self._attention_boost.get("strength", 0.0) or 0.0) * self.attention_boost_decay)
        self._attention_boost["ticks_left"] = ticks_left
        self._attention_boost["strength"] = _round4(strength)
        if ticks_left <= 0 or strength <= 0.0001:
            self._attention_boost = {
                "active": False,
                "strength": 0.0,
                "ticks_left": 0,
                "target_center_hz": _round4(self._audio_focus["center_hz"]),
                "target_bandwidth_octaves": _round4(self._audio_focus["bandwidth_octaves"]),
                "source_action": str(self._attention_boost.get("source_action", "") or ""),
                "window_budget_bonus": 0,
                "focus_budget_bonus": 0,
                "bandwidth_scale": 1.0,
                "focus_gain": 1.0,
            }

    def _focus_bonus(self, *, center_hz: float, focus_center_hz: float, bandwidth_octaves: float, focus_gain: float) -> float:
        log_dist = abs(_safe_log2(center_hz) - _safe_log2(focus_center_hz))
        sigma = max(0.06, float(bandwidth_octaves) * 0.5)
        base = math.exp(-(log_dist * log_dist) / max(1e-6, 2.0 * sigma * sigma))
        return _clamp(base * max(1.0, float(focus_gain)), 0.0, 2.5)

    def _spectral_similarity(self, current: np.ndarray, previous: np.ndarray) -> float:
        if current.size == 0 or previous.size == 0:
            return 0.0
        size = min(int(current.size), int(previous.size))
        if size <= 0:
            return 0.0
        left = np.asarray(current[:size], dtype=np.float32)
        right = np.asarray(previous[:size], dtype=np.float32)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-8:
            return 0.0
        sim = float(np.dot(left, right) / denom)
        return _clamp(sim, 0.0, 1.0)

    def _decay_window_fatigue(self, step_gap: int) -> None:
        if step_gap <= 0:
            return
        decay = float(self.auditory_fatigue_decay)
        next_counts: dict[str, int] = {}
        for key, value in list(self._recent_selected_windows.items()):
            decayed = int(round(max(0.0, float(value) * (decay ** step_gap))))
            if decayed > 0:
                next_counts[str(key)] = decayed
        self._recent_selected_windows = next_counts

    def _audio_cache_key(self, payload: bytes) -> str:
        data = bytes(payload or b"")
        if not data:
            return ""
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _stream_cache_key(
        self,
        stream_frames: bytes,
        *,
        channels: int,
        sampwidth: int,
        framerate: int,
        window_size: int,
        focus_band_count: int,
    ) -> str:
        digest = self._audio_cache_key(stream_frames)
        if not digest:
            return ""
        return f"{digest}:{int(channels)}:{int(sampwidth)}:{int(framerate)}:{int(window_size)}:{int(focus_band_count)}"

    def _get_cached_audio_input(self, key: str) -> dict[str, Any] | None:
        clean = str(key or "")
        if not clean:
            return None
        cached = self._audio_input_cache.pop(clean, None)
        if cached is None:
            return None
        self._audio_input_cache[clean] = cached
        return cached

    def _store_cached_audio_input(self, key: str, payload: dict[str, Any]) -> None:
        clean = str(key or "")
        if not clean:
            return
        self._audio_input_cache[clean] = payload
        while len(self._audio_input_cache) > self._audio_input_cache_limit:
            oldest_key = next(iter(self._audio_input_cache))
            if oldest_key == clean and len(self._audio_input_cache) == 1:
                break
            self._audio_input_cache.pop(oldest_key, None)

    def _get_or_decode_audio_input(self, raw_bytes: bytes) -> dict[str, Any]:
        cache_key = self._audio_cache_key(raw_bytes)
        cached = self._get_cached_audio_input(cache_key)
        if cached is not None:
            return cached
        with wave.open(BytesIO(raw_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        frame_width = max(1, sampwidth * channels)
        payload = {
            "channels": int(channels),
            "sampwidth": int(sampwidth),
            "framerate": int(framerate),
            "frames": bytes(frames),
            "preview_wav_b64": base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else "",
            "preview_audio_bytes_len": int(len(raw_bytes or b"")),
            "preview_duration_ms": _round4((len(frames) / max(1, frame_width)) * 1000.0 / max(1, framerate)),
        }
        self._store_cached_audio_input(cache_key, payload)
        return payload

    def _get_cached_audio_stream(self, key: str) -> dict[str, Any] | None:
        clean = str(key or "")
        if not clean:
            return None
        cached = self._audio_stream_cache.pop(clean, None)
        if cached is None:
            return None
        self._audio_stream_cache[clean] = cached
        return cached

    def _store_cached_audio_stream(self, key: str, payload: dict[str, Any]) -> None:
        clean = str(key or "")
        if not clean:
            return
        self._audio_stream_cache[clean] = payload
        while len(self._audio_stream_cache) > self._audio_stream_cache_limit:
            oldest_key = next(iter(self._audio_stream_cache))
            if oldest_key == clean and len(self._audio_stream_cache) == 1:
                break
            self._audio_stream_cache.pop(oldest_key, None)

    def _decode_mono_samples(self, chunk: bytes, *, channels: int, sampwidth: int) -> np.ndarray:
        if not chunk:
            return np.zeros((0,), dtype=np.float32)
        if sampwidth == 2:
            arr = np.frombuffer(chunk, dtype="<i2")
            if arr.size <= 0:
                return np.zeros((0,), dtype=np.float32)
            if channels > 1:
                frame_count = arr.size // max(1, channels)
                if frame_count <= 0:
                    return np.zeros((0,), dtype=np.float32)
                arr = arr[: frame_count * channels].reshape(frame_count, channels)[:, 0]
            return arr.astype(np.float32, copy=False)
        if sampwidth == 1:
            arr_u8 = np.frombuffer(chunk, dtype=np.uint8)
            if channels > 1:
                frame_count = arr_u8.size // max(1, channels)
                if frame_count <= 0:
                    return np.zeros((0,), dtype=np.float32)
                arr_u8 = arr_u8[: frame_count * channels].reshape(frame_count, channels)[:, 0]
            return (arr_u8.astype(np.float32) - 128.0) * 256.0
        frame_width = max(1, sampwidth * max(1, channels))
        step = max(1, frame_width)
        mono_samples: list[int] = []
        for pos in range(0, len(chunk), step):
            if pos + sampwidth > len(chunk):
                break
            mono_samples.append(int.from_bytes(chunk[pos : pos + sampwidth], byteorder="little", signed=True))
        return np.asarray(mono_samples, dtype=np.float32) if mono_samples else np.zeros((0,), dtype=np.float32)

    def _encode_wav_preview_b64(self, samples: np.ndarray, *, sample_rate: int) -> tuple[str, int]:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return "", 0
        clipped = np.clip(arr, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        buf = BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(max(8000, sample_rate)))
            wav.writeframes(pcm16.tobytes())
        raw = buf.getvalue()
        return base64.b64encode(raw).decode("ascii"), len(raw)

    def _build_proxy_preview_samples(
        self,
        *,
        selected: list[dict[str, Any]],
        window_samples: dict[int, np.ndarray],
        proxy_templates: dict[int, np.ndarray] | None,
        sample_rate: int,
        window_size_samples: int,
        full_window_count: int,
    ) -> np.ndarray:
        total_samples = max(window_size_samples, full_window_count * max(1, window_size_samples))
        proxy = np.zeros((total_samples,), dtype=np.float32)
        if total_samples <= 0 or not selected:
            return proxy
        ordered = sorted(
            [item for item in selected if isinstance(item, dict)],
            key=lambda item: int(item.get("position", 0) or 0),
        )
        templates = dict(proxy_templates or {})
        for item in ordered:
            attrs = dict(item.get("attributes", {}) or {})
            pos = int(item.get("position", 0) or 0)
            proxy_base = np.asarray(templates.get(pos, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
            if proxy_base.size <= 0:
                window_arr = np.asarray(window_samples.get(pos, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
                if window_arr.size <= 0:
                    continue
                proxy_base = window_arr
            if proxy_base.size <= 0:
                continue
            energy = _clamp(float(item.get("energy", 0.0) or 0.0), 0.0, 1.2)
            tonal = _clamp(float(attrs.get("tonal_clarity", 0.0) or 0.0), 0.0, 1.0)
            percussive = _clamp(float(attrs.get("percussive_ratio", 0.0) or 0.0), 0.0, 1.0)
            filtered = proxy_base.copy()
            gain = 0.18 + energy * 0.62
            if percussive > tonal:
                gain *= 0.92 + percussive * 0.18
            filtered = filtered * gain
            start = pos * max(1, window_size_samples)
            end = min(total_samples, start + filtered.size)
            if end <= start:
                continue
            proxy[start:end] += filtered[: end - start]
        peak = float(np.max(np.abs(proxy))) if proxy.size else 0.0
        if peak > 1e-6:
            proxy = proxy / max(1.0, peak / 0.92)
        return np.clip(proxy, -1.0, 1.0).astype(np.float32)

    def build_proxy_audio_for_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        sample_rate: int,
        window_size_samples: int | None = None,
    ) -> np.ndarray:
        picked = [dict(item) for item in (rows or []) if isinstance(item, dict)]
        if not picked:
            return np.zeros((0,), dtype=np.float32)
        ordered = sorted(picked, key=lambda item: int(item.get("position", 0) or 0))
        max_pos = max(int(item.get("position", 0) or 0) for item in ordered)
        base_window_size = int(window_size_samples or 0)
        if base_window_size <= 0:
            template_sizes = []
            for item in ordered:
                attrs = dict(item.get("attributes", {}) or {})
                count = int(attrs.get("window_sample_count", 0) or 0)
                if count > 0:
                    template_sizes.append(count)
            base_window_size = max(template_sizes or [max(1, int(sample_rate * (self.window_ms / 1000.0)))])
        total_samples = max(base_window_size, (max_pos + 1) * max(1, base_window_size))
        proxy = np.zeros((total_samples,), dtype=np.float32)
        for item in ordered:
            attrs = dict(item.get("attributes", {}) or {})
            count = max(1, int(attrs.get("window_sample_count", 0) or base_window_size))
            center_hz = max(
                60.0,
                float(attrs.get("dominant_hz", 0.0) or 0.0)
                or float(attrs.get("spectral_centroid_hz", 0.0) or 0.0)
                or 220.0,
            )
            tonal = _clamp(float(attrs.get("tonal_clarity", 0.0) or 0.0), 0.0, 1.0)
            noisy = _clamp(float(attrs.get("noisiness", 0.0) or 0.0), 0.0, 1.0)
            harmonic = _clamp(float(attrs.get("harmonic_ratio", 0.0) or 0.0), 0.0, 1.0)
            percussive = _clamp(float(attrs.get("percussive_ratio", 0.0) or 0.0), 0.0, 1.0)
            peak_prominence = _clamp(float(attrs.get("peak_prominence", 0.0) or 0.0), 0.0, 1.0)
            energy = _clamp(float(item.get("energy", 0.0) or 0.0), 0.0, 1.25)
            signal_presence = _clamp(float(attrs.get("signal_presence", 0.0) or 0.0), 0.0, 1.0)
            time_axis = np.arange(count, dtype=np.float32) / max(1.0, float(sample_rate))
            template = np.sin(2.0 * math.pi * center_hz * time_axis) * (0.52 + tonal * 0.26)
            if tonal >= 0.08 or harmonic >= 0.08:
                template += np.sin(2.0 * math.pi * center_hz * 2.0 * time_axis) * (0.10 + harmonic * 0.18)
                template += np.sin(2.0 * math.pi * center_hz * 3.0 * time_axis) * (0.04 + harmonic * 0.10)
            if noisy >= 0.08:
                rng = np.random.default_rng(abs(hash((int(center_hz), count, int(energy * 1000)))) % (2**32))
                template += rng.normal(0.0, 1.0, size=count).astype(np.float32) * (0.035 + noisy * 0.07)
            if percussive >= 0.08:
                clicks = np.zeros((count,), dtype=np.float32)
                click_len = max(2, int(count * 0.08))
                clicks[:click_len] = np.linspace(1.0, 0.0, click_len, dtype=np.float32)
                template += clicks * (0.08 + percussive * 0.16)
            attack = max(2, int(count * (0.08 + 0.06 * peak_prominence)))
            release = max(4, int(count * 0.24))
            env = np.ones((count,), dtype=np.float32)
            env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
            env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
            template = template * env * (0.16 + energy * 0.66) * (0.24 + signal_presence * 0.76)
            peak = float(np.max(np.abs(template))) if template.size else 0.0
            if peak > 1e-6:
                template = template / max(1.0, peak / 0.95)
            start = int(item.get("position", 0) or 0) * max(1, base_window_size)
            end = min(total_samples, start + template.size)
            if end > start:
                proxy[start:end] += template[: end - start]
        peak = float(np.max(np.abs(proxy))) if proxy.size else 0.0
        if peak > 1e-6:
            proxy = proxy / max(1.0, peak / 0.92)
        return np.clip(proxy, -1.0, 1.0).astype(np.float32)

    def _build_proxy_window_template(self, samples: np.ndarray, attrs: dict[str, Any], *, sample_rate: int) -> np.ndarray:
        signal = np.asarray(samples, dtype=np.float32).reshape(-1)
        if signal.size <= 0:
            return np.zeros((0,), dtype=np.float32)
        tonal = _clamp(float(attrs.get("tonal_clarity", 0.0) or 0.0), 0.0, 1.0)
        noisy = _clamp(float(attrs.get("noisiness", 0.0) or 0.0), 0.0, 1.0)
        harmonic = _clamp(float(attrs.get("harmonic_ratio", 0.0) or 0.0), 0.0, 1.0)
        percussive = _clamp(float(attrs.get("percussive_ratio", 0.0) or 0.0), 0.0, 1.0)
        center_hz = max(
            60.0,
            float(attrs.get("dominant_hz", 0.0) or 0.0)
            or float(attrs.get("spectral_centroid_hz", 0.0) or 0.0)
            or 220.0,
        )
        bandwidth_ratio = _clamp(float(attrs.get("spectral_bandwidth_ratio", 0.0) or 0.0), 0.0, 1.0)
        sigma_hz = max(70.0, bandwidth_ratio * float(sample_rate) * 0.32 + 90.0)
        filtered = signal.copy()
        if signal.size > 4:
            analysis_window = np.hanning(signal.size).astype(np.float32)
            spectrum = np.fft.rfft(signal * analysis_window)
            freqs = np.fft.rfftfreq(signal.size, d=1.0 / max(1, sample_rate))
            mask = np.exp(-((freqs - center_hz) ** 2) / max(1e-6, 2.0 * sigma_hz * sigma_hz))
            if tonal >= 0.18:
                mask += 0.48 * tonal * np.exp(-((freqs - center_hz * 2.0) ** 2) / max(1e-6, 2.0 * (sigma_hz * 1.15) ** 2))
                mask += 0.26 * harmonic * np.exp(-((freqs - center_hz * 3.0) ** 2) / max(1e-6, 2.0 * (sigma_hz * 1.4) ** 2))
            if noisy >= 0.18:
                mask += 0.14 * noisy
            if percussive >= 0.18:
                rolloff = max(center_hz * 1.15, float(attrs.get("spectral_rolloff_hz", center_hz * 1.8) or center_hz * 1.8))
                mask += 0.22 * percussive * np.clip(freqs / max(1.0, rolloff), 0.0, 1.0)
            filtered = np.fft.irfft(spectrum * np.clip(mask, 0.0, 1.75), n=signal.size).astype(np.float32)
        attack = max(2, int(signal.size * (0.08 + 0.06 * _clamp(float(attrs.get("peak_prominence", 0.0) or 0.0), 0.0, 1.0))))
        release = max(4, int(signal.size * 0.24))
        env = np.ones((filtered.size,), dtype=np.float32)
        if env.size > 0:
            env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
            env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        filtered = filtered * env
        peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
        if peak > 1e-6:
            filtered = filtered / max(1.0, peak / 0.96)
        return filtered.astype(np.float32)

    def _band_mass_from_spectrum(self, spectrum: np.ndarray, band_count: int) -> tuple[np.ndarray, np.ndarray]:
        if spectrum.size <= 0:
            return np.zeros((band_count,), dtype=np.float32), np.linspace(0, 1, band_count + 1, dtype=int)
        band_edges = np.linspace(0, max(1, spectrum.size - 1), band_count + 1, dtype=int)
        band_mass = np.zeros((band_count,), dtype=np.float32)
        for band_index in range(band_count):
            lo = int(band_edges[band_index])
            hi = int(max(lo + 1, band_edges[band_index + 1]))
            if hi > lo:
                band_mass[band_index] = float(np.sum(spectrum[lo:hi]))
        return band_mass, band_edges

    def _spectral_flatness(self, spectrum: np.ndarray) -> float:
        if spectrum.size <= 0:
            return 0.0
        safe = np.maximum(np.asarray(spectrum, dtype=np.float32), 1e-6)
        geometric = float(np.exp(np.mean(np.log(safe))))
        arithmetic = float(np.mean(safe))
        return _clamp(geometric / max(1e-6, arithmetic), 0.0, 1.0)

    def _spectral_bandwidth_ratio(
        self,
        spectrum: np.ndarray,
        *,
        spectral_centroid_hz: float,
        framerate: int,
        sample_count: int,
    ) -> float:
        if spectrum.size <= 1 or sample_count <= 0 or framerate <= 0:
            return 0.0
        freqs = np.arange(spectrum.size, dtype=np.float32) * (float(framerate) / max(1.0, float(sample_count)))
        denom = max(1e-6, float(np.sum(spectrum)))
        variance = float(np.sum(((freqs - float(spectral_centroid_hz)) ** 2) * spectrum) / denom)
        bandwidth_hz = math.sqrt(max(0.0, variance))
        return _clamp(bandwidth_hz / max(1.0, float(framerate) * 0.5), 0.0, 1.0)

    def _spectral_rolloff(
        self,
        spectrum: np.ndarray,
        *,
        framerate: int,
        sample_count: int,
        roll_percent: float = 0.85,
    ) -> tuple[float, float]:
        if spectrum.size <= 0 or sample_count <= 0 or framerate <= 0:
            return 0.0, 0.0
        cumulative = np.cumsum(np.asarray(spectrum, dtype=np.float32))
        total = float(cumulative[-1]) if cumulative.size else 0.0
        if total <= 1e-6:
            return 0.0, 0.0
        target = total * _clamp(float(roll_percent), 0.05, 0.99)
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = max(0, min(int(spectrum.size) - 1, index))
        rolloff_hz = float(index * (float(framerate) / max(1.0, float(sample_count))))
        return rolloff_hz, _clamp(rolloff_hz / max(1.0, float(framerate) * 0.5), 0.0, 1.0)

    def _spectral_contrast(self, spectrum: np.ndarray, band_edges: np.ndarray) -> float:
        if spectrum.size <= 0 or band_edges.size < 2:
            return 0.0
        contrasts: list[float] = []
        for band_index in range(max(0, int(band_edges.size) - 1)):
            lo = int(band_edges[band_index])
            hi = int(max(lo + 2, band_edges[band_index + 1]))
            if hi <= lo or hi > spectrum.size:
                continue
            band = np.asarray(spectrum[lo:hi], dtype=np.float32)
            if band.size < 4:
                continue
            high = float(np.max(band))
            low = float(np.min(band))
            contrast = math.log((high + 1e-6) / max(1e-6, low + 1e-6), 2.0)
            contrasts.append(max(0.0, contrast))
        if not contrasts:
            return 0.0
        return _clamp(float(sum(contrasts) / len(contrasts)) / 8.0, 0.0, 1.0)

    def _audio_profile_tag(self, attrs: dict[str, Any]) -> str:
        tonal = float(attrs.get("tonal_clarity", 0.0) or 0.0)
        noisy = float(attrs.get("noisiness", 0.0) or 0.0)
        percussive = float(attrs.get("percussive_ratio", 0.0) or 0.0)
        harmonic = float(attrs.get("harmonic_ratio", 0.0) or 0.0)
        tag, _ = max(
            [
                ("tonal", tonal),
                ("noisy", noisy),
                ("percussive", percussive),
                ("harmonic", harmonic),
            ],
            key=lambda row: (float(row[1]), str(row[0])),
        )
        return str(tag)

    def _audio_feature_code(self, attrs: dict[str, Any]) -> str:
        dominant_band_index = max(0, int(attrs.get("dominant_band_index", 0) or 0))
        return (
            f"pf{self._audio_profile_tag(attrs)[:2]}"
            f"_tc{_bin_code(float(attrs.get('tonal_clarity', 0.0) or 0.0), 6)}"
            f"_nz{_bin_code(float(attrs.get('noisiness', 0.0) or 0.0), 6)}"
            f"_ps{_bin_code(float(attrs.get('pitch_stability', 0.0) or 0.0), 6)}"
            f"_hr{_bin_code(float(attrs.get('harmonic_ratio', 0.0) or 0.0), 6)}"
            f"_pr{_bin_code(float(attrs.get('percussive_ratio', 0.0) or 0.0), 6)}"
            f"_vp{_bin_code(float(attrs.get('voiced_probability', 0.0) or 0.0), 6)}"
            f"_ct{_bin_code(float(attrs.get('spectral_contrast', 0.0) or 0.0), 6)}"
            f"_fl{_bin_code(float(attrs.get('spectral_flatness', 0.0) or 0.0), 6)}"
            f"_bw{_bin_code(float(attrs.get('spectral_bandwidth_ratio', 0.0) or 0.0), 6)}"
            f"_ro{_bin_code(float(attrs.get('spectral_rolloff_ratio', 0.0) or 0.0), 6)}"
            f"_ce{_bin_code(float(attrs.get('spectral_centroid_ratio', 0.0) or 0.0), 6)}"
            f"_db{dominant_band_index}"
        )

    def _build_audio_memory_feature_item(self, item: dict[str, Any]) -> dict[str, Any]:
        attrs = dict(item.get("attributes", {}) or {})
        coords = dict(item.get("coords", {}) or {})
        feature_code = self._audio_feature_code(attrs)
        structure_strength = max(
            float(attrs.get("tonal_clarity", 0.0) or 0.0),
            float(attrs.get("noisiness", 0.0) or 0.0),
            float(attrs.get("percussive_ratio", 0.0) or 0.0),
            float(attrs.get("pitch_stability", 0.0) or 0.0),
        )
        signal_presence = _clamp(float(attrs.get("signal_presence", 0.0) or 0.0), 0.0, 1.0)
        energy = max(
            0.0,
            float(item.get("energy", 0.0) or 0.0)
            * (0.44 + 0.34 * _clamp(structure_strength, 0.0, 1.0) + 0.22 * signal_presence),
        )
        payload_attrs = dict(attrs)
        payload_attrs["sample_role"] = "memory_feature"
        payload_attrs["memory_feature_code"] = feature_code
        payload_attrs["structure_profile"] = self._audio_profile_tag(attrs)
        payload_attrs["structure_strength"] = _round4(structure_strength)
        return {
            "sa_label": f"audio::mem::{feature_code}",
            "display_text": f"听觉特征[{feature_code}]",
            "energy": _round4(energy),
            "position": int(item.get("position", 0) or 0),
            "source_type": str(item.get("source_type", "audio_input") or "audio_input"),
            "sa_kind": "audio_memory_feature_unit",
            "coords": coords,
            "attributes": payload_attrs,
            "channel": "hearing",
        }

    def _summarize_audio_features(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        feature_keys = [
            "tonal_clarity",
            "noisiness",
            "pitch_stability",
            "harmonic_ratio",
            "percussive_ratio",
            "voiced_probability",
            "spectral_contrast",
            "spectral_flatness",
            "spectral_bandwidth_ratio",
            "spectral_rolloff_ratio",
            "spectral_centroid_ratio",
            "onset_strength",
            "novelty",
            "peak_prominence",
        ]
        total_weight = 0.0
        accum = {key: 0.0 for key in feature_keys}
        band_mass = 0.0
        band_weighted = 0.0
        freq_hz = 0.0
        freq_weighted = 0.0
        signal_presence_weighted = 0.0
        signal_window_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            attrs = dict(item.get("attributes", {}) or {})
            signal_presence = _clamp(float(attrs.get("signal_presence", 0.0) or 0.0), 0.0, 1.0)
            weight = max(0.0, float(item.get("energy", 0.0) or 0.0)) * max(0.0, signal_presence)
            if weight <= 1e-6:
                continue
            total_weight += weight
            signal_presence_weighted += weight * signal_presence
            if signal_presence >= 0.05:
                signal_window_count += 1
            for key in feature_keys:
                accum[key] += weight * float(attrs.get(key, 0.0) or 0.0)
            dominant_band = max(0.0, float(attrs.get("dominant_band_index", 0.0) or 0.0))
            band_weighted += weight * dominant_band
            band_mass += weight
            dominant_hz = max(0.0, float(attrs.get("dominant_hz", 0.0) or 0.0))
            freq_weighted += weight * dominant_hz
            freq_hz += weight
        if total_weight <= 0.0:
            return {"window_count": 0, "dominant_profile": "none"}
        summary = {
            key: _round4(accum[key] / total_weight)
            for key in feature_keys
        }
        summary["dominant_band_index"] = int(round(band_weighted / max(1e-6, band_mass))) if band_mass > 0.0 else 0
        summary["dominant_hz"] = _round4(freq_weighted / max(1e-6, freq_hz)) if freq_hz > 0.0 else 0.0
        summary["signal_presence"] = _round4(signal_presence_weighted / total_weight)
        summary["window_count"] = int(signal_window_count)
        summary["dominant_profile"] = self._audio_profile_tag(summary)
        summary["memory_feature_code"] = self._audio_feature_code(summary)
        return summary

    def _build_stream_window_analysis(
        self,
        *,
        stream_frames: bytes,
        channels: int,
        sampwidth: int,
        framerate: int,
        window_size: int,
        focus_band_count: int,
    ) -> dict[str, Any]:
        total_len = len(stream_frames)
        full_window_count = total_len // window_size if window_size > 0 else 0
        leftover = stream_frames[full_window_count * window_size :] if full_window_count > 0 else stream_frames
        window_rows: list[dict[str, Any]] = []
        window_samples: dict[int, np.ndarray] = {}
        proxy_templates: dict[int, np.ndarray] = {}
        window_size_samples = max(1, window_size // max(1, sampwidth * channels))
        for index in range(full_window_count):
            start = index * window_size
            chunk = stream_frames[start : start + window_size]
            rms = audioop.rms(chunk, sampwidth)
            samples_arr = self._decode_mono_samples(chunk, channels=channels, sampwidth=sampwidth)
            sample_count = int(samples_arr.size)
            zero_cross = 0
            if sample_count > 1:
                prev_sign = np.signbit(samples_arr[:-1])
                next_sign = np.signbit(samples_arr[1:])
                zero_cross = int(np.count_nonzero(prev_sign != next_sign))
            if samples_arr.size:
                copied = np.asarray(samples_arr, dtype=np.float32).copy()
                window_samples[int(index)] = copied
                abs_samples = np.abs(copied)
                sample_peak_norm = float(np.max(abs_samples)) / 32768.0 if abs_samples.size else 0.0
                sample_abs_mean_norm = float(np.mean(abs_samples)) / 32768.0 if abs_samples.size else 0.0
                analysis_window = np.hanning(copied.size) if copied.size > 1 else np.ones((copied.size,), dtype=np.float32)
                spectrum = np.abs(np.fft.rfft(copied * analysis_window))
                spectrum_total = float(np.sum(spectrum))
                band_mass_arr, band_edges = self._band_mass_from_spectrum(spectrum, focus_band_count)
                dominant_band_index = int(np.argmax(band_mass_arr)) if band_mass_arr.size else 0
                dominant_bin = int(np.argmax(spectrum)) if spectrum.size else 0
                dominant_ratio = float((dominant_bin / max(1, spectrum.size - 1))) if spectrum.size > 1 else 0.0
                dominant_hz = float(dominant_bin * framerate / max(1, copied.size))
                spectral_centroid_hz = (
                    float(np.sum(np.arange(spectrum.size, dtype=np.float32) * spectrum) / max(1e-6, spectrum_total) * framerate / max(1, copied.size))
                    if spectrum.size
                    else 0.0
                )
                spectral_centroid_ratio = _clamp(spectral_centroid_hz / max(1.0, float(framerate) * 0.5), 0.0, 1.0)
                spectral_flatness = self._spectral_flatness(spectrum)
                spectral_contrast = self._spectral_contrast(spectrum, band_edges)
                spectral_bandwidth_ratio = self._spectral_bandwidth_ratio(
                    spectrum,
                    spectral_centroid_hz=spectral_centroid_hz,
                    framerate=framerate,
                    sample_count=sample_count,
                )
                spectral_rolloff_hz, spectral_rolloff_ratio = self._spectral_rolloff(
                    spectrum,
                    framerate=framerate,
                    sample_count=sample_count,
                )
                peak_prominence = _clamp(float(np.max(spectrum)) / max(1e-6, spectrum_total) * 12.0, 0.0, 1.0)
                low_split = max(1, focus_band_count // 4)
                high_split = max(2, (focus_band_count * 3) // 4)
                low_band = float(np.sum(band_mass_arr[:low_split]) / max(1e-6, spectrum_total))
                mid_band = float(np.sum(band_mass_arr[low_split:high_split]) / max(1e-6, spectrum_total))
                high_band = float(np.sum(band_mass_arr[high_split:]) / max(1e-6, spectrum_total))
                norm_band_mass = band_mass_arr / max(1e-6, float(np.sum(band_mass_arr)))
            else:
                sample_peak_norm = 0.0
                sample_abs_mean_norm = 0.0
                spectrum_total = 0.0
                dominant_band_index = 0
                dominant_ratio = 0.0
                dominant_hz = 0.0
                spectral_centroid_hz = 0.0
                spectral_centroid_ratio = 0.0
                spectral_flatness = 0.0
                spectral_contrast = 0.0
                spectral_bandwidth_ratio = 0.0
                spectral_rolloff_hz = 0.0
                spectral_rolloff_ratio = 0.0
                peak_prominence = 0.0
                low_band = 0.0
                mid_band = 0.0
                high_band = 0.0
                norm_band_mass = np.zeros((focus_band_count,), dtype=np.float32)
            row = {
                "index": int(index),
                "rms": float(rms) / 32768.0,
                "zero_cross_rate": float(zero_cross) / max(1, sample_count),
                "sample_count": int(sample_count),
                "sample_peak_norm": _round4(sample_peak_norm),
                "sample_abs_mean_norm": _round4(sample_abs_mean_norm),
                "dominant_band_index": int(dominant_band_index),
                "dominant_bin_ratio": _round4(dominant_ratio),
                "dominant_hz": _round4(dominant_hz),
                "spectral_centroid_hz": _round4(spectral_centroid_hz),
                "spectral_centroid_ratio": _round4(spectral_centroid_ratio),
                "spectral_flatness": _round4(spectral_flatness),
                "spectral_contrast": _round4(spectral_contrast),
                "spectral_bandwidth_ratio": _round4(spectral_bandwidth_ratio),
                "spectral_rolloff_hz": _round4(spectral_rolloff_hz),
                "spectral_rolloff_ratio": _round4(spectral_rolloff_ratio),
                "peak_prominence": _round4(peak_prominence),
                "low_band": _round4(low_band),
                "mid_band": _round4(mid_band),
                "high_band": _round4(high_band),
                "spectrum_total": _round4(spectrum_total),
                "norm_band_mass": [float(value) for value in norm_band_mass.tolist()],
            }
            window_rows.append(row)
            if sample_count > 0 and int(index) in window_samples:
                proxy_templates[int(index)] = self._build_proxy_window_template(window_samples[int(index)], row, sample_rate=framerate)
        return {
            "full_window_count": int(full_window_count),
            "leftover": bytes(leftover),
            "window_rows": window_rows,
            "window_samples": window_samples,
            "proxy_templates": proxy_templates,
            "window_size_samples": int(window_size_samples),
        }

    def ingest_wav_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "audio_input") -> dict[str, Any]:
        self._sensor_tick += 1
        step_gap = max(0, int(tick_index) - self._stream_chunk_index - 1) if self._stream_chunk_index >= 0 else 0
        self._decay_window_fatigue(step_gap + 1)
        self._stream_chunk_index += 1
        decoded = self._get_or_decode_audio_input(raw_bytes)
        channels = int(decoded.get("channels", 1) or 1)
        sampwidth = int(decoded.get("sampwidth", 2) or 2)
        framerate = int(decoded.get("framerate", 16000) or 16000)
        frames = bytes(decoded.get("frames", b"") or b"")
        frame_width = max(1, sampwidth * channels)
        window_size = max(frame_width, int(framerate * (self.window_ms / 1000.0)) * frame_width)
        stream_frames = self._carry_frames + frames

        profile = self._effective_sampling_profile()
        focus_snapshot = self.audio_focus_snapshot()
        boost = dict(profile.get("boost", {}) or {})
        focus_center_hz = float(boost.get("target_center_hz", focus_snapshot["center_hz"]) or focus_snapshot["center_hz"])
        bandwidth_octaves = max(0.08, float(profile.get("bandwidth_octaves", focus_snapshot["bandwidth_octaves"]) or focus_snapshot["bandwidth_octaves"]))
        focus_gain = max(1.0, float(profile.get("focus_gain", 1.0) or 1.0))
        effective_window_budget = max(1, int(profile.get("window_budget", self.window_budget) or self.window_budget))
        effective_focus_band_count = max(1, int(profile.get("focus_band_count", self.focus_band_count) or self.focus_band_count))

        stream_cache_key = self._stream_cache_key(
            stream_frames,
            channels=channels,
            sampwidth=sampwidth,
            framerate=framerate,
            window_size=window_size,
            focus_band_count=effective_focus_band_count,
        )
        stream_cached = self._get_cached_audio_stream(stream_cache_key)
        if stream_cached is None:
            stream_cached = self._build_stream_window_analysis(
                stream_frames=stream_frames,
                channels=channels,
                sampwidth=sampwidth,
                framerate=framerate,
                window_size=window_size,
                focus_band_count=effective_focus_band_count,
            )
            self._store_cached_audio_stream(stream_cache_key, stream_cached)

        full_window_count = int(stream_cached.get("full_window_count", 0) or 0)
        leftover = bytes(stream_cached.get("leftover", b"") or b"")
        window_rows = list(stream_cached.get("window_rows", []) or [])
        base_window_samples = dict(stream_cached.get("window_samples", {}) or {})
        base_proxy_templates = dict(stream_cached.get("proxy_templates", {}) or {})

        windows: list[dict[str, Any]] = []
        rms_values: list[float] = []
        zero_cross_values: list[float] = []
        focus_priority_samples: list[dict[str, Any]] = []
        memory_write_samples: list[dict[str, Any]] = []
        global_structure_samples: list[dict[str, Any]] = []
        stream_band_mass = np.zeros((effective_focus_band_count,), dtype=np.float32)
        window_samples: dict[int, np.ndarray] = {int(key): np.asarray(value, dtype=np.float32) for key, value in base_window_samples.items()}
        proxy_templates: dict[int, np.ndarray] = {int(key): np.asarray(value, dtype=np.float32) for key, value in base_proxy_templates.items()}

        for row in window_rows:
            index = int(row.get("index", 0) or 0)
            rms_norm = float(row.get("rms", 0.0) or 0.0)
            zero_cross_rate = float(row.get("zero_cross_rate", 0.0) or 0.0)
            sample_count = int(row.get("sample_count", 0) or 0)
            sample_peak_norm = float(row.get("sample_peak_norm", 0.0) or 0.0)
            sample_abs_mean_norm = float(row.get("sample_abs_mean_norm", 0.0) or 0.0)
            dominant_band_index = int(row.get("dominant_band_index", 0) or 0)
            dominant_ratio = float(row.get("dominant_bin_ratio", 0.0) or 0.0)
            dominant_hz = float(row.get("dominant_hz", 0.0) or 0.0)
            spectral_centroid_hz = float(row.get("spectral_centroid_hz", 0.0) or 0.0)
            spectral_centroid_ratio = float(row.get("spectral_centroid_ratio", 0.0) or 0.0)
            spectral_flatness = float(row.get("spectral_flatness", 0.0) or 0.0)
            spectral_contrast = float(row.get("spectral_contrast", 0.0) or 0.0)
            spectral_bandwidth_ratio = float(row.get("spectral_bandwidth_ratio", 0.0) or 0.0)
            spectral_rolloff_hz = float(row.get("spectral_rolloff_hz", 0.0) or 0.0)
            spectral_rolloff_ratio = float(row.get("spectral_rolloff_ratio", 0.0) or 0.0)
            peak_prominence = float(row.get("peak_prominence", 0.0) or 0.0)
            low_band = float(row.get("low_band", 0.0) or 0.0)
            mid_band = float(row.get("mid_band", 0.0) or 0.0)
            high_band = float(row.get("high_band", 0.0) or 0.0)
            spectrum_total = float(row.get("spectrum_total", 0.0) or 0.0)
            norm_band_mass = np.asarray((row.get("norm_band_mass", []) or []), dtype=np.float32)
            if norm_band_mass.size != effective_focus_band_count:
                resized = np.zeros((effective_focus_band_count,), dtype=np.float32)
                if norm_band_mass.size > 0:
                    resized[: min(effective_focus_band_count, norm_band_mass.size)] = norm_band_mass[: min(effective_focus_band_count, norm_band_mass.size)]
                norm_band_mass = resized

            prev_state = self._prev_window_state.get(str(index), {})
            prev_rms = float(prev_state.get("rms", rms_norm) or rms_norm)
            prev_peak_hz = float(prev_state.get("dominant_hz", dominant_hz) or dominant_hz)
            prev_band_mass = np.asarray((prev_state.get("band_mass", []) or []), dtype=np.float32)
            delta = min(1.0, abs(rms_norm - prev_rms) * 6.0)
            spectral_similarity = self._spectral_similarity(norm_band_mass, prev_band_mass)
            stable_band = spectral_similarity >= (1.0 - self.static_dedup_band_similarity_threshold)
            static_like = delta < self.static_dedup_delta_threshold and stable_band
            zero_cross_norm = _clamp(zero_cross_rate * 6.0, 0.0, 1.0)
            fatigue_hits = int(self._recent_selected_windows.get(str(index), 0) or 0)
            fatigue_penalty = 0.0
            if static_like:
                fatigue_penalty = min(
                    self.static_dedup_max_suppression,
                    min(self.auditory_fatigue_max, fatigue_hits * self.auditory_fatigue_step) * 0.65 + 0.25,
                )
            focus_bonus = self._focus_bonus(
                center_hz=max(40.0, dominant_hz or spectral_centroid_hz or focus_center_hz),
                focus_center_hz=focus_center_hz,
                bandwidth_octaves=bandwidth_octaves,
                focus_gain=focus_gain,
            )
            onset_strength = _clamp(delta * 0.72 + abs(dominant_hz - prev_peak_hz) / max(80.0, framerate * 0.08), 0.0, 1.0)
            novelty = _clamp((1.0 - spectral_similarity) * 0.7 + delta * 0.3, 0.0, 1.0)
            voiced_probability = _clamp(
                0.38 * (1.0 - spectral_flatness)
                + 0.24 * spectral_contrast
                + 0.22 * peak_prominence
                + 0.16 * (1.0 - zero_cross_norm),
                0.0,
                1.0,
            )
            hz_stability = 1.0
            if dominant_hz > 0.0 and prev_peak_hz > 0.0:
                hz_stability = math.exp(
                    -(
                        abs(_safe_log2(max(40.0, dominant_hz)) - _safe_log2(max(40.0, prev_peak_hz))) ** 2
                    )
                    / max(1e-6, 2.0 * 0.42 * 0.42)
                )
            pitch_stability = _clamp(
                0.46 * hz_stability
                + 0.24 * spectral_similarity
                + 0.18 * voiced_probability
                + 0.12 * (1.0 - delta),
                0.0,
                1.0,
            )
            harmonic_ratio = _clamp(
                0.34 * voiced_probability
                + 0.22 * (1.0 - spectral_flatness)
                + 0.18 * spectral_contrast
                + 0.14 * (1.0 - zero_cross_norm)
                + 0.12 * peak_prominence,
                0.0,
                1.0,
            )
            percussive_ratio = _clamp(
                0.32 * onset_strength
                + 0.20 * zero_cross_norm
                + 0.18 * spectral_flatness
                + 0.14 * high_band
                + 0.10 * delta
                + 0.10 * (1.0 - pitch_stability),
                0.0,
                1.0,
            )
            tonal_clarity = _clamp(
                0.42 * harmonic_ratio
                + 0.24 * spectral_contrast
                + 0.18 * (1.0 - spectral_flatness)
                + 0.16 * pitch_stability,
                0.0,
                1.0,
            )
            noisiness = _clamp(
                0.46 * spectral_flatness
                + 0.18 * high_band
                + 0.16 * zero_cross_norm
                + 0.12 * (1.0 - spectral_contrast)
                + 0.08 * (1.0 - harmonic_ratio),
                0.0,
                1.0,
            )
            signal_presence = _clamp(
                0.42 * _clamp(sample_abs_mean_norm * 14.0, 0.0, 1.0)
                + 0.24 * _clamp(sample_peak_norm * 2.8, 0.0, 1.0)
                + 0.18 * _clamp(rms_norm * 12.0, 0.0, 1.0)
                + 0.10 * _clamp(peak_prominence * 1.2, 0.0, 1.0)
                + 0.06 * (1.0 if spectrum_total > 1e-6 and (dominant_hz > 0.0 or spectral_centroid_hz > 0.0) else 0.0),
                0.0,
                1.0,
            )
            signal_gate = _clamp(
                (0.74 * signal_presence + 0.26 * max(tonal_clarity, percussive_ratio, voiced_probability)) ** 0.82,
                0.0,
                1.0,
            )
            rms_level = min(1.0, rms_norm * (32768.0 / 20000.0))
            energy = (
                rms_level * 0.34
                + min(1.0, zero_cross_rate * 4.0) * 0.06
                + max(0.0, low_band * 0.08 + mid_band * 0.18 + high_band * 0.08)
                + onset_strength * 0.18
                + novelty * 0.16
                + focus_bonus * 0.22
                + tonal_clarity * 0.06
                + percussive_ratio * 0.04
                - fatigue_penalty * 0.45
            ) * signal_gate
            raw_priority = (
                novelty * 0.28
                + onset_strength * 0.24
                + focus_bonus * 0.30
                + min(1.0, rms_norm * 2.5) * 0.18
                + max(tonal_clarity, percussive_ratio, noisiness) * 0.08
                - fatigue_penalty * 0.55
            ) * (0.12 + 0.88 * signal_gate)
            focus_priority = (
                focus_bonus * 0.46
                + novelty * 0.18
                + onset_strength * 0.18
                + min(1.0, rms_norm * 2.2) * 0.12
                + tonal_clarity * 0.10
                + (0.06 if mid_band >= max(low_band, high_band) else 0.0)
            ) * (0.10 + 0.90 * signal_gate)
            sample_reason = "audio_focus" if focus_bonus >= 0.95 else "onset_probe" if onset_strength >= 0.28 else "ambient_scan"
            selected_energy = _round4(max(0.0, energy))
            band_low_ratio = float(dominant_band_index / max(1, effective_focus_band_count))
            band_high_ratio = float((dominant_band_index + 1) / max(1, effective_focus_band_count))
            band_center_ratio = float((band_low_ratio + band_high_ratio) * 0.5)
            band_center_hz = max(40.0, (framerate * 0.5) * band_center_ratio)
            item = {
                "sa_label": f"audio::win_{index}",
                "display_text": f"听窗[{index}]",
                "energy": selected_energy,
                "position": index,
                "source_type": source_type,
                "sa_kind": "audio_window_unit",
                "coords": {
                    "time_window_index": int(index),
                    "freq_center_hz": _round4(band_center_hz),
                    "freq_center_ratio": _round4(band_center_ratio),
                    "freq_low_ratio": _round4(band_low_ratio),
                    "freq_high_ratio": _round4(band_high_ratio),
                    "octave_distance_from_focus": _round4(abs(_safe_log2(max(40.0, band_center_hz)) - _safe_log2(focus_center_hz))),
                },
                "attributes": {
                    "rms": _round4(rms_norm),
                    "zero_cross_rate": _round4(zero_cross_rate),
                    "delta": _round4(delta),
                    "low_band": _round4(low_band),
                    "mid_band": _round4(mid_band),
                    "high_band": _round4(high_band),
                    "dominant_bin_ratio": _round4(dominant_ratio),
                    "dominant_hz": _round4(dominant_hz),
                    "spectral_centroid_hz": _round4(spectral_centroid_hz),
                    "spectral_centroid_ratio": _round4(spectral_centroid_ratio),
                    "spectral_bandwidth_ratio": _round4(spectral_bandwidth_ratio),
                    "spectral_rolloff_hz": _round4(spectral_rolloff_hz),
                    "spectral_rolloff_ratio": _round4(spectral_rolloff_ratio),
                    "spectral_flatness": _round4(spectral_flatness),
                    "spectral_contrast": _round4(spectral_contrast),
                    "peak_prominence": _round4(peak_prominence),
                    "dominant_band_index": int(dominant_band_index),
                    "focus_bonus": _round4(focus_bonus),
                    "onset_strength": _round4(onset_strength),
                    "novelty": _round4(novelty),
                    "voiced_probability": _round4(voiced_probability),
                    "pitch_stability": _round4(pitch_stability),
                    "harmonic_ratio": _round4(harmonic_ratio),
                    "percussive_ratio": _round4(percussive_ratio),
                    "tonal_clarity": _round4(tonal_clarity),
                    "noisiness": _round4(noisiness),
                    "sample_peak_norm": _round4(sample_peak_norm),
                    "sample_abs_mean_norm": _round4(sample_abs_mean_norm),
                    "signal_presence": _round4(signal_presence),
                    "signal_gate": _round4(signal_gate),
                    "spectral_similarity": _round4(spectral_similarity),
                    "fatigue_penalty": _round4(fatigue_penalty),
                    "raw_priority": _round4(raw_priority),
                    "focus_priority": _round4(focus_priority),
                    "sample_reason": sample_reason,
                    "sample_role": "audio_window",
                    "window_sample_count": int(sample_count),
                    "static_like": bool(static_like),
                },
                "channel": "hearing",
                "_norm_band_mass": [float(value) for value in norm_band_mass.tolist()],
            }
            windows.append(item)
            rms_values.append(rms_norm)
            zero_cross_values.append(zero_cross_rate)
            if stream_band_mass.size == norm_band_mass.size:
                stream_band_mass += norm_band_mass

        def _signal_presence_of(row: dict[str, Any]) -> float:
            return _clamp(float(((row.get("attributes", {}) or {}).get("signal_presence", 0.0) or 0.0)), 0.0, 1.0)

        def _weighted_audio_priority(row: dict[str, Any], key_name: str) -> float:
            attrs = dict(row.get("attributes", {}) or {})
            return float(attrs.get(key_name, 0.0) or 0.0) * (0.18 + 0.82 * _signal_presence_of(row))

        windows.sort(
            key=lambda item: (
                -_weighted_audio_priority(item, "raw_priority"),
                -float(item.get("energy", 0.0) or 0.0),
                int(item.get("position", 0) or 0),
            )
        )
        preferred = [
            item
            for item in windows
            if _signal_presence_of(item) >= 0.05
        ]
        fallback = [item for item in windows if item not in preferred]
        selected = (preferred + fallback)[:effective_window_budget]
        selected_labels = {str(item.get("sa_label", "") or "") for item in selected}

        focus_ranked = sorted(
            windows,
            key=lambda item: (
                -_weighted_audio_priority(item, "focus_priority"),
                -float(item.get("energy", 0.0) or 0.0),
                int(item.get("position", 0) or 0),
            ),
        )
        focus_priority_samples = [
            self._clone_with_reason(item, sample_reason="audio_focus_memory")
            for item in focus_ranked[: min(len(focus_ranked), effective_focus_band_count)]
        ]
        memory_write_samples = [
            self._clone_with_reason(item, sample_reason=str((item.get("attributes", {}) or {}).get("sample_reason", "audio_window") or "audio_window"))
            for item in selected[: min(len(selected), max(4, effective_focus_band_count // 2))]
        ]
        structured_memory_samples = [
            self._build_audio_memory_feature_item(item)
            for item in focus_ranked[: min(len(focus_ranked), max(4, effective_focus_band_count // 2))]
        ]
        memory_write_samples.extend(structured_memory_samples)

        feature_summary = self._summarize_audio_features(selected)

        if stream_band_mass.size:
            stream_band_norm = stream_band_mass / max(1e-6, float(np.sum(stream_band_mass)))
            peak_band = int(np.argmax(stream_band_norm)) if stream_band_norm.size else 0
            peak_center_ratio = (peak_band + 0.5) / max(1, stream_band_norm.size)
            peak_center_hz = max(40.0, (framerate * 0.5) * peak_center_ratio)
            global_structure_samples.append(
                {
                    "sa_label": f"audio::global_band_{peak_band}",
                    "display_text": f"听觉全局特征[band::{peak_band}]",
                    "energy": _round4(max(0.05, float(stream_band_norm[peak_band]) if stream_band_norm.size > peak_band else 0.05)),
                    "position": int(peak_band),
                    "source_type": source_type,
                    "sa_kind": "audio_global_feature_unit",
                    "coords": {
                        "freq_center_hz": _round4(peak_center_hz),
                        "freq_center_ratio": _round4(peak_center_ratio),
                    },
                    "attributes": {
                        "sample_role": "global_structure",
                        "peak_band_index": int(peak_band),
                        "peak_band_mass": _round4(float(stream_band_norm[peak_band]) if stream_band_norm.size > peak_band else 0.0),
                        "focus_distance_octaves": _round4(abs(_safe_log2(peak_center_hz) - _safe_log2(focus_center_hz))),
                    },
                    "channel": "hearing",
                }
            )
            if int(feature_summary.get("window_count", 0) or 0) > 0:
                summary_attrs = dict(feature_summary)
                summary_attrs["sample_role"] = "global_structure"
                summary_attrs["global_feature_code"] = str(feature_summary.get("memory_feature_code", "") or "")
                summary_attrs["focus_distance_octaves"] = _round4(abs(_safe_log2(max(40.0, float(feature_summary.get("dominant_hz", 0.0) or 0.0) or peak_center_hz)) - _safe_log2(focus_center_hz)))
                global_structure_samples.append(
                    {
                        "sa_label": f"audio::global::{summary_attrs['global_feature_code']}",
                        "display_text": f"听觉全局结构[{summary_attrs['global_feature_code']}]",
                        "energy": _round4(
                            max(
                                0.06,
                                0.42 * float(feature_summary.get("tonal_clarity", 0.0) or 0.0)
                                + 0.24 * float(feature_summary.get("percussive_ratio", 0.0) or 0.0)
                                + 0.18 * float(feature_summary.get("noisiness", 0.0) or 0.0)
                                + 0.16 * float(feature_summary.get("novelty", 0.0) or 0.0),
                            )
                        ),
                        "position": int(peak_band),
                        "source_type": source_type,
                        "sa_kind": "audio_global_feature_unit",
                        "coords": {
                            "freq_center_hz": _round4(max(40.0, float(feature_summary.get("dominant_hz", 0.0) or 0.0) or peak_center_hz)),
                            "freq_center_ratio": _round4(peak_center_ratio),
                        },
                        "attributes": summary_attrs,
                        "channel": "hearing",
                    }
                )

        next_prev_state: dict[str, dict[str, Any]] = {}
        next_selected_windows: dict[str, int] = {}
        for item in windows:
            position = str(int(item.get("position", 0) or 0))
            attrs = dict(item.get("attributes", {}) or {})
            next_prev_state[position] = {
                "rms": float(attrs.get("rms", 0.0) or 0.0),
                "delta": float(attrs.get("delta", 0.0) or 0.0),
                "dominant_hz": float(attrs.get("dominant_hz", 0.0) or 0.0),
                "band_mass": [float(value) for value in np.asarray(item.get("_norm_band_mass", []), dtype=np.float32).tolist()],
            }
            prior_hits = int(self._recent_selected_windows.get(position, 0) or 0)
            if str(item.get("sa_label", "") or "") in selected_labels:
                boosted = min(self.auditory_fatigue_max, prior_hits + self.auditory_fatigue_step * 2.0)
                next_selected_windows[position] = max(0, int(round(boosted * 10.0)))
            else:
                cooled = max(0.0, prior_hits * self.auditory_fatigue_decay - 1.0)
                if cooled > 0.0:
                    next_selected_windows[position] = int(round(cooled))
        if windows:
            strongest = max(windows, key=lambda row: float(row.get("energy", 0.0) or 0.0))
            next_prev_state["__global__"] = {
                "peak_hz": float(((strongest.get("attributes", {}) or {}).get("dominant_hz", focus_center_hz) or focus_center_hz)),
            }
        self._prev_window_state = next_prev_state
        self._recent_selected_windows = next_selected_windows
        self._carry_frames = leftover[-window_size:] if window_size > 0 else b""

        current_mean_energy = sum(float(item.get("energy", 0.0) or 0.0) for item in selected) / max(1, len(selected))
        self._energy_history.append(float(current_mean_energy))
        stream_rms_mean = sum(rms_values) / max(1, len(rms_values)) if rms_values else 0.0
        stream_zcr_mean = sum(zero_cross_values) / max(1, len(zero_cross_values)) if zero_cross_values else 0.0
        window_size_samples = int(stream_cached.get("window_size_samples", max(1, window_size // max(1, frame_width))) or max(1, window_size // max(1, frame_width)))
        proxy_preview = self._build_proxy_preview_samples(
            selected=selected,
            window_samples=window_samples,
            proxy_templates=proxy_templates,
            sample_rate=framerate,
            window_size_samples=window_size_samples,
            full_window_count=full_window_count,
        )
        proxy_preview_wav_b64, proxy_preview_bytes_len = self._encode_wav_preview_b64(proxy_preview, sample_rate=framerate)

        packet = {
            "schema_id": "hearing_sensor_packet/v2",
            "schema_version": "2.0",
            "sensor_name": "hearing_sensor_v1",
            "tick_index": int(tick_index),
            "sensor_tick": self._sensor_tick,
            "source_type": source_type,
            "audio_meta": {"channels": channels, "sample_width": sampwidth, "framerate": framerate},
            "preview_wav_b64": str(decoded.get("preview_wav_b64", "") or ""),
            "preview_audio_bytes_len": int(decoded.get("preview_audio_bytes_len", 0) or 0),
            "proxy_preview_wav_b64": proxy_preview_wav_b64,
            "proxy_preview_audio_bytes_len": int(proxy_preview_bytes_len),
            "preview_duration_ms": float(decoded.get("preview_duration_ms", 0.0) or 0.0),
            "window_ms": self.window_ms,
            "budget_used": len(selected),
            "raw_window_budget": int(effective_window_budget),
            "focus_band_budget": int(effective_focus_band_count),
            "focus_priority_samples": focus_priority_samples,
            "memory_write_samples": memory_write_samples,
            "global_structure_samples": global_structure_samples,
            "windows": selected,
            "feature_summary": feature_summary,
            "audio_focus": self.audio_focus_snapshot(),
            "attention_boost": self.attention_boost_snapshot(),
            "stream_state": {
                "chunk_index": self._stream_chunk_index,
                "carry_frame_bytes": len(self._carry_frames),
                "full_window_count": full_window_count,
                "stream_rms_mean": _round4(stream_rms_mean),
                "stream_zero_cross_mean": _round4(stream_zcr_mean),
                "selected_energy_mean": _round4(current_mean_energy),
                "energy_history_mean": _round4(sum(self._energy_history) / max(1, len(self._energy_history))),
                "effective_window_budget": int(effective_window_budget),
                "effective_focus_band_budget": int(effective_focus_band_count),
                "focus_center_hz": _round4(focus_center_hz),
                "focus_bandwidth_octaves": _round4(bandwidth_octaves),
                "attention_mode": str(profile.get("attention_mode", "background") or "background"),
            },
        }
        self._decay_attention_boost()
        return packet

    def _clone_with_reason(self, item: dict[str, Any], *, sample_reason: str) -> dict[str, Any]:
        cloned = dict(item)
        if "coords" in cloned:
            cloned["coords"] = dict(item.get("coords", {}) or {})
        cloned["attributes"] = dict(item.get("attributes", {}) or {})
        cloned["attributes"]["sample_reason"] = str(sample_reason or "")
        return cloned

    def export_payload(self) -> dict[str, Any]:
        return {
            "sensor_tick": self._sensor_tick,
            "window_ms": self.window_ms,
            "stream_chunk_index": self._stream_chunk_index,
            "carry_frames_b64": self._carry_frames.hex(),
            "prev_window_state": self._prev_window_state,
            "recent_selected_windows": self._recent_selected_windows,
            "energy_history": list(self._energy_history),
            "audio_focus": self.audio_focus_snapshot(),
            "attention_boost": self.attention_boost_snapshot(),
            "attention_mode": str(self._attention_mode or "background"),
            "focus_band_count": int(self.focus_band_count),
            "focus_bandwidth_octaves": _round4(self.focus_bandwidth_octaves),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._sensor_tick = int(payload.get("sensor_tick", 0) or 0)
        if "window_ms" in payload:
            self.window_ms = max(5, int(payload.get("window_ms", self.window_ms) or self.window_ms))
        self._stream_chunk_index = int(payload.get("stream_chunk_index", -1) or -1)
        carry_hex = str(payload.get("carry_frames_b64", "") or "")
        try:
            self._carry_frames = bytes.fromhex(carry_hex) if carry_hex else b""
        except Exception:
            self._carry_frames = b""
        prev_window_state = payload.get("prev_window_state", {}) or {}
        self._prev_window_state = {
            str(key or ""): dict(value)
            for key, value in prev_window_state.items()
            if isinstance(value, dict) and str(key or "")
        }
        recent_selected = payload.get("recent_selected_windows", {}) or {}
        self._recent_selected_windows = {str(key or ""): max(0, int(value or 0)) for key, value in recent_selected.items() if str(key or "")}
        self._energy_history = deque((float(item or 0.0) for item in (payload.get("energy_history", []) or [])), maxlen=24)
        if isinstance(payload.get("audio_focus"), dict):
            focus = payload.get("audio_focus", {}) or {}
            self.move_audio_focus(
                float(focus.get("center_hz", self._audio_focus["center_hz"]) or self._audio_focus["center_hz"]),
                bandwidth_octaves=float(focus.get("bandwidth_octaves", self._audio_focus["bandwidth_octaves"]) or self._audio_focus["bandwidth_octaves"]),
            )
        if isinstance(payload.get("attention_boost"), dict):
            boost = payload.get("attention_boost", {}) or {}
            self._attention_boost = {
                "active": bool(boost.get("active", False)),
                "strength": _round4(float(boost.get("strength", 0.0) or 0.0)),
                "ticks_left": max(0, int(boost.get("ticks_left", 0) or 0)),
                "target_center_hz": _round4(float(boost.get("target_center_hz", self._audio_focus["center_hz"]) or self._audio_focus["center_hz"])),
                "target_bandwidth_octaves": _round4(float(boost.get("target_bandwidth_octaves", self._audio_focus["bandwidth_octaves"]) or self._audio_focus["bandwidth_octaves"])),
                "source_action": str(boost.get("source_action", "") or ""),
                "window_budget_bonus": max(0, int(boost.get("window_budget_bonus", 0) or 0)),
                "focus_budget_bonus": max(0, int(boost.get("focus_budget_bonus", 0) or 0)),
                "bandwidth_scale": _round4(float(boost.get("bandwidth_scale", 1.0) or 1.0)),
                "focus_gain": _round4(float(boost.get("focus_gain", 1.0) or 1.0)),
            }
        self._attention_mode = str(payload.get("attention_mode", self._attention_mode) or self._attention_mode)
        if "focus_band_count" in payload:
            self.focus_band_count = max(3, int(payload.get("focus_band_count", self.focus_band_count) or self.focus_band_count))
        if "focus_bandwidth_octaves" in payload:
            self.focus_bandwidth_octaves = max(0.08, float(payload.get("focus_bandwidth_octaves", self.focus_bandwidth_octaves) or self.focus_bandwidth_octaves))
