from __future__ import annotations

import io
import math
import wave

import numpy as np
from sensors.reconstruction_payload import make_reconstruction_payload, payload_summary_vector


def _round4(value: float) -> float:
    return round(float(value), 4)


def _safe_vector(values: list[float], *, cap: int) -> list[float]:
    return [_round4(float(value or 0.0)) for value in values[: max(1, int(cap))]]


class NativeAudioNumericSensor:
    """
    Fixed-budget numeric audio sensor for WAV input.

    It emits spectrum/band/rhythm/pitch/event SAs. Raw waveform bytes are
    consumed by the sensor, but inner_audio carries only numeric reconstruction
    hints so observatory playback cannot shortcut through the original input.
    """

    def __init__(self, *, max_samples: int = 32768, band_count: int = 12) -> None:
        self.max_samples = max(1024, int(max_samples))
        self.band_count = max(6, int(band_count))

    def ingest_wav_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "audio_input", focus_state: dict | None = None) -> dict:
        focus = self._normalize_focus_state(focus_state)
        samples, sample_rate, duration_ms = self._read_wav(raw_bytes)
        if samples.size > self.max_samples:
            idx = np.linspace(0, samples.size - 1, self.max_samples).astype(np.int64)
            samples = samples[idx]
        features = self._features(samples, sample_rate, focus_state=focus)
        state_items = self._build_state_items(features=features, tick_index=tick_index, source_type=source_type, focus_state=focus)
        inner_audio = {
            "schema_id": "inner_audio_numeric/v1",
            "current_bands": [f"audio::global::band_{idx}" for idx in range(min(8, self.band_count))],
            "primary_peaks": ["audio_event::current"],
            "recall_stream": [],
            "prediction_stream": [],
            "preview_asset_ref": {
                "reconstruction_basis": "state_pool_numeric_channels",
                "raw_preview_payload": False,
                "preview_duration_ms": _round4(duration_ms),
                "sample_rate": int(sample_rate),
                "sensor_focus_state": focus,
            },
            "feature_summary": {
                "dominant_hz": _round4(float(features["dominant_hz"])),
                "spectral_centroid_hz": _round4(float(features["spectral_centroid_hz"])),
                "rms": _round4(float(features["rms"])),
                "onset_strength": _round4(float(features["onset_strength"])),
                "sample_rate": int(sample_rate),
                "speech_like_reconstruction": dict(features.get("speech_like_reconstruction", {}) or {}),
            },
            "focus_reconstruction": {
                "schema_id": "audio_focus_reconstruction/v1",
                "reconstruction_basis": "state_pool_numeric_channels",
                "sensor_focus_state": focus,
                "payloads": dict(features.get("reconstruction_payloads", {}) or {}),
            },
            "event_reconstruction": {
                "schema_id": "reconstruction_payload_bundle/v1",
                "modality": "audio",
                "scope": "focus",
                "sensor_focus_state": focus,
                "channels": dict(features.get("reconstruction_payloads", {}) or {}),
            },
            "energy_summary": {"state_item_count": len(state_items), "duration_ms": _round4(duration_ms)},
        }
        return {
            "packet": {
                "schema_id": "audio_numeric_packet/v1",
                "tick_index": int(tick_index),
                "source_type": source_type,
                "sample_rate": int(sample_rate),
                "duration_ms": _round4(duration_ms),
                "focus_state": focus,
            },
            "state_items": state_items,
            "inner_audio": inner_audio,
        }

    def _read_wav(self, raw_bytes: bytes) -> tuple[np.ndarray, int, float]:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wav:
            channels = max(1, int(wav.getnchannels()))
            sample_width = int(wav.getsampwidth())
            sample_rate = int(wav.getframerate())
            frame_count = int(wav.getnframes())
            data = wav.readframes(frame_count)
        if sample_width == 1:
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sample_width == 2:
            arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            arr = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported wav sample width: {sample_width}")
        if channels > 1 and arr.size >= channels:
            arr = arr[: (arr.size // channels) * channels].reshape(-1, channels).mean(axis=1)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        duration_ms = (float(arr.size) / max(1.0, float(sample_rate))) * 1000.0
        return arr, sample_rate, duration_ms

    def _features(self, samples: np.ndarray, sample_rate: int, focus_state: dict | None = None) -> dict:
        focus = self._normalize_focus_state(focus_state)
        if samples.size <= 0:
            samples = np.zeros((1,), dtype=np.float32)
        samples = np.clip(samples.astype(np.float32), -1.0, 1.0)
        rms = float(np.sqrt(np.mean(samples * samples)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(samples).astype(np.float32))))) if samples.size > 1 else 0.0
        window = np.hanning(samples.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(samples * window))
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / max(1, int(sample_rate)))
        power = spectrum * spectrum
        power_sum = max(1e-9, float(power.sum()))
        centroid = float((freqs * power).sum() / power_sum)
        dominant_idx = int(np.argmax(power)) if power.size else 0
        dominant_hz = float(freqs[dominant_idx]) if freqs.size else 0.0
        clarity = float(power[dominant_idx] / power_sum) if power.size else 0.0
        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, power_sum * 0.85)) if cumulative.size else 0
        rolloff = float(freqs[min(rolloff_idx, max(0, freqs.size - 1))]) if freqs.size else 0.0
        band_vector = self._band_vector(freqs, power, sample_rate, focus_state=focus)
        rhythm_vector, onset_strength = self._rhythm_vector(samples)
        reconstruction_payloads = self._focus_reconstruction_payloads(samples=samples, sample_rate=sample_rate, focus_state=focus)
        waveform_payload = dict(reconstruction_payloads.get("audio.focus.waveform_slice", {}) or {})
        waveform_points = len(list(waveform_payload.get("payload_values", []) or []))
        speech_like_reconstruction = {
            "schema_id": "audio_speech_like_reconstruction_diagnostics/v1",
            "sample_count": int(samples.size),
            "sample_rate": int(sample_rate),
            "waveform_payload_points": int(waveform_points),
            "waveform_coverage_ratio": _round4(min(1.0, float(waveform_points) / max(1.0, float(samples.size)))),
            "policy": "state_pool_numeric_payload_not_raw_audio_asset",
        }
        focus_band = self._focus_band_vector(sample_rate=sample_rate, focus_state=focus)
        pitch_vector = [
            dominant_hz / max(1.0, float(sample_rate) * 0.5),
            clarity,
            centroid / max(1.0, float(sample_rate) * 0.5),
            rolloff / max(1.0, float(sample_rate) * 0.5),
            zcr,
            rms,
        ]
        event_vector = band_vector[:6] + rhythm_vector[:4] + pitch_vector[:4]
        return {
            "audio.spectrum": band_vector,
            "audio.band": band_vector,
            "audio.rhythm": rhythm_vector,
            "audio.pitch": _safe_vector(pitch_vector, cap=8),
            "audio.event": _safe_vector(event_vector, cap=20),
            "audio.focus_band": focus_band,
            "dominant_hz": dominant_hz,
            "spectral_centroid_hz": centroid,
            "rms": rms,
            "onset_strength": onset_strength,
            "sample_rate": int(sample_rate),
            "reconstruction_payloads": reconstruction_payloads,
            "speech_like_reconstruction": speech_like_reconstruction,
        }

    def _band_vector(self, freqs: np.ndarray, power: np.ndarray, sample_rate: int, focus_state: dict | None = None) -> list[float]:
        if freqs.size <= 1 or power.size <= 1:
            return [0.0] * self.band_count
        max_hz = max(200.0, min(float(sample_rate) * 0.5, 8000.0))
        edges = np.geomspace(40.0, max_hz, self.band_count + 1)
        rows = []
        total = max(1e-9, float(power.sum()))
        for idx in range(self.band_count):
            mask = (freqs >= edges[idx]) & (freqs < edges[idx + 1])
            value = float(power[mask].sum()) / total if mask.any() else 0.0
            center = math.sqrt(float(edges[idx]) * float(edges[idx + 1]))
            value *= 1.0 + self._focus_gain_for_hz(center_hz=center, focus_state=focus_state) * 0.35
            rows.append(value)
        row_total = max(1e-9, float(sum(rows)))
        rows = [float(value) / row_total for value in rows]
        return _safe_vector(rows, cap=self.band_count)

    def _rhythm_vector(self, samples: np.ndarray) -> tuple[list[float], float]:
        frame_count = 32
        if samples.size < frame_count:
            padded = np.pad(samples, (0, frame_count - samples.size))
        else:
            padded = samples[: (samples.size // frame_count) * frame_count]
        frames = padded.reshape(frame_count, -1) if padded.size >= frame_count else padded.reshape(frame_count, 1)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        diffs = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
        onset_strength = float(diffs.mean())
        if float(rms.max()) > 1e-9:
            rms_norm = rms / float(rms.max())
        else:
            rms_norm = rms
        autocorr = np.correlate(rms_norm - rms_norm.mean(), rms_norm - rms_norm.mean(), mode="full")[len(rms_norm) - 1 :]
        if autocorr.size > 3 and float(np.max(np.abs(autocorr[1:]))) > 1e-9:
            period_idx = int(np.argmax(autocorr[1: min(16, autocorr.size)]) + 1)
            periodicity = float(autocorr[period_idx] / max(1e-9, autocorr[0]))
        else:
            period_idx = 0
            periodicity = 0.0
        vector = [
            float(rms.mean()),
            float(rms.std()),
            onset_strength,
            period_idx / 16.0,
            max(0.0, periodicity),
            float((diffs > max(0.02, onset_strength)).mean()),
        ]
        return _safe_vector(vector, cap=8), onset_strength

    def _focus_reconstruction_payloads(self, *, samples: np.ndarray, sample_rate: int, focus_state: dict | None = None) -> dict[str, dict]:
        focus = self._normalize_focus_state(focus_state)
        clean = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if clean.size <= 0:
            clean = np.zeros((1,), dtype=np.float32)
        envelope = self._envelope(clean, frame_count=128)
        waveform = self._waveform_slice(clean, max_points=self._waveform_slice_budget(clean.size, focus_state=focus))
        stft_mag, stft_phase = self._stft_patch(clean, freq_bins=32, time_bins=24)
        focus_precision = self._focus_precision(sample_rate=sample_rate, focus_state=focus)
        pitch = self._pitch_contour(clean, sample_rate=sample_rate, frame_count=32)
        onsets = self._onset_events(envelope, limit=16)
        transient = self._transient_curve(clean, frame_count=128)
        harmonic_noise = self._harmonic_noise_curve(clean, sample_rate=sample_rate, frame_count=32)
        duration_ms = (float(clean.size) / max(1.0, float(sample_rate))) * 1000.0
        rms = float(np.sqrt(np.mean(clean * clean)))
        return {
            "audio.focus.waveform_slice": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.waveform_slice",
                scope="focus",
                fidelity_level="near_lossless_numeric_slice" if clean.size <= len(waveform) else "focused_resampled_numeric_slice",
                summary_vector=[rms, float(np.max(np.abs(clean))), float(clean.mean()), float(clean.std()), duration_ms / 1000.0],
                payload_shape=[int(len(waveform))],
                payload_values=waveform.tolist(),
                sampling_precision=min(1.0, float(len(waveform)) / max(1.0, float(clean.size))) * focus_precision,
                payload_limit=len(waveform),
            ),
            "audio.focus.envelope": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.envelope",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._curve_summary(envelope),
                payload_shape=[int(len(envelope))],
                payload_values=envelope.tolist(),
                sampling_precision=min(1.0, 0.78 * focus_precision),
                payload_limit=256,
            ),
            "audio.focus.stft_magnitude": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.stft_magnitude",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._matrix_summary(stft_mag),
                payload_shape=[int(stft_mag.shape[0]), int(stft_mag.shape[1])],
                payload_values=stft_mag.tolist(),
                sampling_precision=min(1.0, 0.68 * focus_precision),
                payload_limit=2048,
            ),
            "audio.focus.stft_phase": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.stft_phase",
                scope="focus",
                fidelity_level="phase_like",
                summary_vector=self._matrix_summary(stft_phase),
                payload_shape=[int(stft_phase.shape[0]), int(stft_phase.shape[1])],
                payload_values=stft_phase.tolist(),
                sampling_precision=min(1.0, 0.68 * focus_precision),
                payload_limit=2048,
            ),
            "audio.focus.pitch_contour": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.pitch_contour",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._curve_summary(pitch),
                payload_shape=[int(len(pitch))],
                payload_values=pitch.tolist(),
                sampling_precision=min(1.0, 0.62 * focus_precision),
                payload_limit=128,
            ),
            "audio.focus.onset_events": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.onset_events",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._curve_summary(np.asarray([row[1] for row in onsets], dtype=np.float32)),
                payload_shape=[int(len(onsets)), 2],
                payload_values=onsets,
                sampling_precision=min(1.0, 0.72 * focus_precision),
                payload_limit=64,
            ),
            "audio.focus.transient": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.transient",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._curve_summary(transient),
                payload_shape=[int(len(transient))],
                payload_values=transient.tolist(),
                sampling_precision=min(1.0, 0.66 * focus_precision),
                payload_limit=256,
            ),
            "audio.focus.harmonic_noise": make_reconstruction_payload(
                modality="audio",
                channel="audio.focus.harmonic_noise",
                scope="focus",
                fidelity_level="mid",
                summary_vector=self._curve_summary(harmonic_noise.reshape(-1)),
                payload_shape=[int(harmonic_noise.shape[0]), int(harmonic_noise.shape[1])],
                payload_values=harmonic_noise.tolist(),
                sampling_precision=min(1.0, 0.56 * focus_precision),
                payload_limit=256,
            ),
        }

    def _waveform_slice_budget(self, sample_count: int, focus_state: dict | None = None) -> int:
        """
        Allocate a bounded but speech-friendly numeric slice.

        The observatory must not replay raw audio assets as AP's inner voice.
        For short speech-like inputs, AP can still carry a denser focused
        waveform payload, analogous to hearing a small attended window more
        clearly. Longer inputs remain capped to keep the tick budget bounded.
        """

        count = max(1, int(sample_count))
        focus = self._normalize_focus_state(focus_state)
        width_hz = float(focus.get("width_hz", 2400.0) or 2400.0)
        focus_bonus = 1.0 + max(0.0, min(1.0, (3600.0 - width_hz) / 3600.0)) * 0.45
        if count <= 24000:
            return min(count, int(18000 * focus_bonus))
        if count <= 72000:
            return min(count, int(24000 * focus_bonus))
        return min(count, int(12000 * focus_bonus))

    def _waveform_slice(self, samples: np.ndarray, *, max_points: int) -> np.ndarray:
        cap = max(64, int(max_points))
        if samples.size <= cap:
            return samples.astype(np.float32)
        idx = np.linspace(0, samples.size - 1, cap).astype(np.int64)
        return samples[idx].astype(np.float32)

    def _envelope(self, samples: np.ndarray, *, frame_count: int) -> np.ndarray:
        count = max(8, int(frame_count))
        padded = self._pad_to_frames(samples, count)
        frames = padded.reshape(count, -1)
        env = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
        peak = float(env.max()) if env.size else 0.0
        return (env / peak).astype(np.float32) if peak > 1e-9 else env

    def _transient_curve(self, samples: np.ndarray, *, frame_count: int) -> np.ndarray:
        envelope = self._envelope(samples, frame_count=frame_count)
        diffs = np.maximum(0.0, np.diff(envelope, prepend=envelope[0])).astype(np.float32)
        peak = float(diffs.max()) if diffs.size else 0.0
        return (diffs / peak).astype(np.float32) if peak > 1e-9 else diffs

    def _stft_patch(self, samples: np.ndarray, *, freq_bins: int, time_bins: int) -> tuple[np.ndarray, np.ndarray]:
        bins = max(8, int(freq_bins))
        frames = max(4, int(time_bins))
        padded = self._pad_to_frames(samples, frames)
        chunks = padded.reshape(frames, -1)
        window = np.hanning(chunks.shape[1]).astype(np.float32) if chunks.shape[1] > 1 else np.ones((1,), dtype=np.float32)
        mag_rows = []
        phase_rows = []
        for chunk in chunks:
            spec = np.fft.rfft(chunk * window)
            mag = np.abs(spec).astype(np.float32)
            phase = (np.angle(spec).astype(np.float32) / math.pi).astype(np.float32)
            mag_rows.append(self._resample_curve(mag, bins))
            phase_rows.append(self._resample_curve(phase, bins))
        mag_arr = np.asarray(mag_rows, dtype=np.float32)
        mag_peak = float(mag_arr.max()) if mag_arr.size else 0.0
        if mag_peak > 1e-9:
            mag_arr = mag_arr / mag_peak
        phase_arr = np.asarray(phase_rows, dtype=np.float32)
        return mag_arr.astype(np.float32), phase_arr.astype(np.float32)

    def _pitch_contour(self, samples: np.ndarray, *, sample_rate: int, frame_count: int) -> np.ndarray:
        count = max(4, int(frame_count))
        padded = self._pad_to_frames(samples, count)
        frames = padded.reshape(count, -1)
        rows = []
        for chunk in frames:
            if chunk.size < 4 or float(np.max(np.abs(chunk))) <= 1e-6:
                rows.append(0.0)
                continue
            window = np.hanning(chunk.size).astype(np.float32)
            power = np.abs(np.fft.rfft(chunk * window)) ** 2
            if power.size <= 1:
                rows.append(0.0)
                continue
            power[0] = 0.0
            idx = int(np.argmax(power))
            hz = idx * float(sample_rate) / max(1.0, float(chunk.size))
            rows.append(float(hz) / max(1.0, float(sample_rate) * 0.5))
        return np.asarray(rows, dtype=np.float32)

    def _onset_events(self, envelope: np.ndarray, *, limit: int) -> list[list[float]]:
        if envelope.size <= 1:
            return []
        diffs = np.maximum(0.0, np.diff(envelope, prepend=envelope[0]))
        threshold = max(0.08, float(diffs.mean() + diffs.std() * 0.45))
        candidates = [(idx, float(value)) for idx, value in enumerate(diffs.tolist()) if float(value) >= threshold]
        candidates.sort(key=lambda row: (-float(row[1]), int(row[0])))
        selected = sorted(candidates[: max(1, int(limit))], key=lambda row: row[0])
        denom = max(1.0, float(envelope.size - 1))
        return [[_round4(float(idx) / denom), _round4(value)] for idx, value in selected]

    def _harmonic_noise_curve(self, samples: np.ndarray, *, sample_rate: int, frame_count: int) -> np.ndarray:
        count = max(4, int(frame_count))
        padded = self._pad_to_frames(samples, count)
        frames = padded.reshape(count, -1)
        rows = []
        for chunk in frames:
            window = np.hanning(chunk.size).astype(np.float32) if chunk.size > 1 else np.ones((1,), dtype=np.float32)
            power = np.abs(np.fft.rfft(chunk * window)) ** 2
            total = max(1e-9, float(power.sum()))
            if power.size <= 2:
                rows.append([0.0, 1.0])
                continue
            power[0] = 0.0
            peak_idx = int(np.argmax(power))
            peak = float(power[peak_idx])
            left = max(0, peak_idx - 1)
            right = min(power.size, peak_idx + 2)
            harmonic = float(power[left:right].sum()) / total
            noise = max(0.0, 1.0 - harmonic)
            rows.append([_round4(harmonic), _round4(noise)])
        return np.asarray(rows, dtype=np.float32)

    def _pad_to_frames(self, samples: np.ndarray, frame_count: int) -> np.ndarray:
        count = max(1, int(frame_count))
        if samples.size <= 0:
            samples = np.zeros((1,), dtype=np.float32)
        frame_len = max(1, int(math.ceil(samples.size / float(count))))
        target = frame_len * count
        if samples.size < target:
            return np.pad(samples, (0, target - samples.size)).astype(np.float32)
        return samples[:target].astype(np.float32)

    def _resample_curve(self, values: np.ndarray, points: int) -> np.ndarray:
        count = max(1, int(points))
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return np.zeros((count,), dtype=np.float32)
        if arr.size == count:
            return arr.astype(np.float32)
        src = np.linspace(0.0, 1.0, arr.size)
        dst = np.linspace(0.0, 1.0, count)
        return np.interp(dst, src, arr).astype(np.float32)

    def _curve_summary(self, values: np.ndarray) -> list[float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return [0.0] * 8
        diffs = np.diff(arr) if arr.size > 1 else np.zeros((1,), dtype=np.float32)
        return _safe_vector(
            [
                float(arr.mean()),
                float(arr.std()),
                float(arr.min()),
                float(arr.max()),
                float(np.mean(np.maximum(0.0, diffs))),
                float(np.mean(np.minimum(0.0, diffs))),
                float(np.mean(np.abs(diffs))),
                float(arr[-1] - arr[0]) if arr.size > 1 else 0.0,
            ],
            cap=12,
        )

    def _matrix_summary(self, values: np.ndarray) -> list[float]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size <= 0:
            return [0.0] * 12
        if arr.ndim == 1:
            return self._curve_summary(arr)
        time_mean = arr.mean(axis=1)
        freq_mean = arr.mean(axis=0)
        return _safe_vector(self._curve_summary(time_mean) + self._curve_summary(freq_mean), cap=24)

    def _build_state_items(self, *, features: dict, tick_index: int, source_type: str, focus_state: dict | None = None) -> list[dict]:
        focus = self._normalize_focus_state(focus_state)
        payloads = dict(features.get("reconstruction_payloads", {}) or {})
        items = [
            {
                "sa_label": "audio_event::current",
                "display_text": "audio event",
                "source_type": "audio_numeric",
                "family": "audio_event",
                "position": 0,
                "real_energy": _round4(max(0.08, min(1.2, float(features.get("rms", 0.0) or 0.0) * 4.0 + float(features.get("onset_strength", 0.0) or 0.0) * 6.0))),
                "numeric_features": {
                    "audio.spectrum": features["audio.spectrum"],
                    "audio.band": features["audio.band"],
                    "audio.rhythm": features["audio.rhythm"],
                    "audio.pitch": features["audio.pitch"],
                    "audio.event": features["audio.event"],
                    "audio.focus_band": features.get("audio.focus_band", []),
                    **{channel: payload_summary_vector(payload) for channel, payload in payloads.items()},
                },
                "reconstruction_payload": {
                    "schema_id": "reconstruction_payload_bundle/v1",
                    "modality": "audio",
                    "scope": "focus",
                    "channels": payloads,
                },
                "anchor_meta": {
                    "channel": "audio.event",
                    "tick_index": int(tick_index),
                    "source_type": source_type,
                    "learnable_handle": True,
                    "dominant_hz": _round4(float(features.get("dominant_hz", 0.0) or 0.0)),
                    "reconstruction_channels": sorted(payloads),
                    "sampling_focus": {
                        "schema_id": "audio_focused_band_sampling/v1",
                        "center_hz": focus["center_hz"],
                        "width_hz": focus["width_hz"],
                        "precision": _round4(self._focus_precision(sample_rate=int(features.get("sample_rate", 16000) or 16000), focus_state=focus)),
                    },
                    "sensor_focus_state": focus,
                },
            }
        ]
        for idx, channel in enumerate(("audio.spectrum", "audio.band", "audio.rhythm", "audio.pitch", "audio.focus_band"), start=1):
            short = channel.split(".")[-1]
            items.append(
                {
                    "sa_label": f"audio::global::{short}",
                    "display_text": f"audio global {short}",
                    "source_type": "audio_numeric",
                    "family": "audio_channel",
                    "position": idx,
                    "real_energy": _round4(max(0.05, items[0]["real_energy"] * 0.65)),
                    "numeric_features": {channel: features[channel]},
                    "anchor_meta": {
                        "channel": channel,
                        "tick_index": int(tick_index),
                        "source_type": source_type,
                        "sensor_focus_state": focus,
                    },
                }
            )
        for channel, payload in sorted(payloads.items()):
            short = channel.split(".")[-1]
            items.append(
                {
                    "sa_label": f"audio::focus::{short}",
                    "display_text": f"audio focus {short}",
                    "source_type": "audio_numeric",
                    "family": "audio_channel",
                    "position": len(items),
                    "real_energy": _round4(max(0.04, items[0]["real_energy"] * 0.72)),
                    "numeric_features": {channel: payload_summary_vector(payload)},
                    "reconstruction_payload": payload,
                    "anchor_meta": {
                        "channel": channel,
                        "tick_index": int(tick_index),
                        "source_type": source_type,
                        "feature_scope": "focus_reconstruction_payload",
                        "sensor_focus_state": focus,
                    },
                }
            )
        return items

    def _normalize_focus_state(self, focus_state: dict | None) -> dict:
        focus = dict(focus_state or {})
        return {
            "center_hz": _round4(max(40.0, min(8000.0, float(focus.get("center_hz", 1000.0) or 1000.0)))),
            "width_hz": _round4(max(120.0, min(8000.0, float(focus.get("width_hz", 2400.0) or 2400.0)))),
            "last_target": str(focus.get("last_target", "") or ""),
            "reconstruction_policy": "focused_frequency_numeric_sampling",
        }

    def _focus_gain_for_hz(self, *, center_hz: float, focus_state: dict | None = None) -> float:
        focus = self._normalize_focus_state(focus_state)
        half_width = max(60.0, float(focus.get("width_hz", 2400.0) or 2400.0) * 0.5)
        distance = abs(float(center_hz) - float(focus.get("center_hz", 1000.0) or 1000.0))
        return max(0.0, min(1.0, 1.0 - distance / half_width))

    def _focus_precision(self, *, sample_rate: int, focus_state: dict | None = None) -> float:
        focus = self._normalize_focus_state(focus_state)
        nyquist = max(1.0, float(sample_rate) * 0.5)
        width_ratio = max(0.02, min(1.0, float(focus.get("width_hz", 2400.0) or 2400.0) / nyquist))
        return max(0.45, min(1.0, 1.05 - width_ratio * 0.45))

    def _focus_band_vector(self, *, sample_rate: int, focus_state: dict | None = None) -> list[float]:
        focus = self._normalize_focus_state(focus_state)
        nyquist = max(1.0, float(sample_rate) * 0.5)
        center = float(focus.get("center_hz", 1000.0) or 1000.0)
        width = float(focus.get("width_hz", 2400.0) or 2400.0)
        low = max(0.0, center - width * 0.5)
        high = min(nyquist, center + width * 0.5)
        return _safe_vector(
            [
                center / nyquist,
                width / nyquist,
                low / nyquist,
                high / nyquist,
                self._focus_precision(sample_rate=sample_rate, focus_state=focus),
            ],
            cap=8,
        )
