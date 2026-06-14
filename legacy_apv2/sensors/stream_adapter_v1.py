# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import struct
import wave
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from sensors.video_decoder_v1 import open_video_decoder_v1


def _round4(value: float) -> float:
    return round(float(value), 4)


class BaseRealtimeSourceV1:
    def __init__(self, *, source_kind: str, source_type: str, total_items: int | None = None, realtime: bool = False) -> None:
        self.source_kind = str(source_kind or "unknown")
        self.source_type = str(source_type or source_kind or "unknown")
        self.total_items = total_items
        self.realtime = bool(realtime)
        self.index = 0
        self.closed = False
        self.exhausted = False
        self.unavailable = False
        self.last_error = ""

    def next_item(self) -> dict[str, Any] | None:
        if self.closed or self.exhausted or self.unavailable:
            return None
        try:
            item = self._next_item_impl()
        except Exception as exc:
            self.last_error = str(exc)
            self.exhausted = True
            raise
        if item is None:
            self.exhausted = True
            return None
        clean = dict(item)
        clean.setdefault("source_type", self.source_type)
        clean.setdefault(
            "stream_source_meta",
            {
                "source_kind": self.source_kind,
                "source_type": self.source_type,
                "item_index": self.index,
                "total_items": self.total_items if self.total_items is not None else 0,
                "realtime": self.realtime,
            },
        )
        self.index += 1
        return clean

    def _next_item_impl(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_type": self.source_type,
            "index": self.index,
            "total_items": self.total_items,
            "realtime": self.realtime,
            "closed": self.closed,
            "exhausted": self.exhausted,
            "unavailable": self.unavailable,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self.closed = True


class SequenceRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(self, *, source_kind: str, source_type: str, items: list[dict[str, Any]], realtime: bool = False) -> None:
        super().__init__(source_kind=source_kind, source_type=source_type, total_items=len(items), realtime=realtime)
        self._items = [dict(item) for item in items]

    def _next_item_impl(self) -> dict[str, Any] | None:
        if self.index >= len(self._items):
            return None
        return dict(self._items[self.index])


class UnavailableRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(self, *, source_kind: str, source_type: str, reason: str) -> None:
        super().__init__(source_kind=source_kind, source_type=source_type, total_items=0, realtime=True)
        self.unavailable = True
        self.last_error = str(reason or "unavailable")

    def _next_item_impl(self) -> dict[str, Any] | None:
        return None


class AudioWavRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(
        self,
        *,
        raw_bytes: bytes,
        tick_window_ms: int,
        source_type: str = "audio_stream",
    ) -> None:
        self._buffer = BytesIO(raw_bytes)
        self._wav: wave.Wave_read | None = None
        self.channels = 0
        self.sampwidth = 0
        self.framerate = 0
        self.frames_per_tick = 0
        self.frame_width = 0
        try:
            self._wav = wave.open(self._buffer, "rb")
            self.channels = int(self._wav.getnchannels())
            self.sampwidth = int(self._wav.getsampwidth())
            self.framerate = int(self._wav.getframerate())
            self.frame_width = max(1, self.sampwidth * self.channels)
            self.frames_per_tick = max(1, int(self.framerate * (max(5, int(tick_window_ms)) / 1000.0)))
            total_frames = int(self._wav.getnframes())
            total_items = int(math.ceil(total_frames / max(1, self.frames_per_tick)))
            super().__init__(source_kind="audio_file", source_type=source_type, total_items=total_items, realtime=False)
        except Exception as exc:
            super().__init__(source_kind="audio_file", source_type=source_type, total_items=0, realtime=False)
            self.unavailable = True
            self.last_error = f"audio_wav_open_failed:{exc}"

    def _next_item_impl(self) -> dict[str, Any] | None:
        if self._wav is None:
            return None
        chunk = self._wav.readframes(self.frames_per_tick)
        if not chunk:
            return None
        payload = BytesIO()
        with wave.open(payload, "wb") as out:
            out.setnchannels(self.channels)
            out.setsampwidth(self.sampwidth)
            out.setframerate(self.framerate)
            out.writeframes(chunk)
        chunk_size = max(1, self.frames_per_tick * self.frame_width)
        return {
            "audio_bytes": payload.getvalue(),
            "source_type": self.source_type,
            "stream_chunk_meta": {
                "chunk_index": self.index,
                "chunk_count": int(self.total_items or 0),
                "tick_window_ms": int(round(self.frames_per_tick * 1000.0 / max(1, self.framerate))),
                "framerate": self.framerate,
                "channels": self.channels,
                "sample_width": self.sampwidth,
                "coverage_ratio": _round4(len(chunk) / max(1, chunk_size)),
                "decode_backend": "wave_bytes_lazy",
            },
        }

    def close(self) -> None:
        try:
            if self._wav is not None:
                self._wav.close()
        finally:
            self._wav = None
            self._buffer.close()
            super().close()


class OpenCVVideoRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(
        self,
        *,
        raw_bytes: bytes,
        tick_fps: float | None = None,
        frame_stride: int | None = None,
        max_frames: int | None = None,
        source_type: str = "video_stream",
        file_hint: str = "",
        suffix_hint: str = ".mp4",
    ) -> None:
        self._decoder = None
        self._decoder_attempts: list[dict[str, Any]] = []
        self._native_fps = 25.0
        self._stride = 1
        self._frame_limit = max(1, int(max_frames or 0)) if max_frames else 0
        self._total_frames = 0
        self._current_index = 0
        self._sampled_index = 0
        self._suffix = str(suffix_hint or ".mp4")
        self._file_hint = str(file_hint or "")
        try:
            self._decoder, self._decoder_attempts = open_video_decoder_v1(
                raw_bytes=raw_bytes,
                file_hint=self._file_hint,
                suffix_hint=self._suffix,
            )
            decoder_status = self._decoder.status()
            self._native_fps = float(decoder_status.get("native_fps", 25.0) or 25.0)
            if self._native_fps <= 0.0:
                self._native_fps = 25.0
            self._stride = max(1, int(frame_stride or 0))
            if self._stride <= 1 and tick_fps and float(tick_fps) > 0:
                self._stride = max(1, int(round(self._native_fps / max(0.1, float(tick_fps)))))
            self._total_frames = int(decoder_status.get("total_frames", 0) or 0)
            total_items = 0
            if self._total_frames > 0:
                total_items = int(math.ceil(self._total_frames / max(1, self._stride)))
                if self._frame_limit:
                    total_items = min(total_items, self._frame_limit)
            elif self._frame_limit:
                total_items = self._frame_limit
            super().__init__(
                source_kind="video_file",
                source_type=source_type,
                total_items=(total_items or None),
                realtime=False,
            )
        except Exception as exc:
            super().__init__(source_kind="video_file", source_type=source_type, total_items=0, realtime=False)
            self.unavailable = True
            self.last_error = f"video_source_open_failed:{exc}"
            self.close()

    def _next_item_impl(self) -> dict[str, Any] | None:
        if self._decoder is None:
            return None
        if self._frame_limit and self._sampled_index >= self._frame_limit:
            return None
        while True:
            decoded = self._decoder.read_frame_png()
            if decoded is None:
                return None
            current_index, encoded_png, decode_meta = decoded
            self._current_index = int(current_index) + 1
            if current_index % self._stride != 0:
                continue
            sampled_index = self._sampled_index
            self._sampled_index += 1
            if self._frame_limit and self._sampled_index >= self._frame_limit:
                self.exhausted = True
            return {
                "image_bytes": bytes(encoded_png),
                "source_type": self.source_type,
                "stream_frame_meta": {
                    "frame_index": sampled_index,
                    "sampled_from_frame_index": current_index,
                    "frame_count_estimate": max(self._total_frames, sampled_index + 1),
                    "native_fps": _round4(self._native_fps),
                    "sample_stride": self._stride,
                    "decode_backend": str(decode_meta.get("decode_backend", "") or "unknown_video_decoder"),
                    "decode_mode": str(decode_meta.get("decode_mode", "") or ""),
                    "temp_suffix": str(decode_meta.get("temp_suffix", "") or ""),
                    "file_hint": str(decode_meta.get("file_hint", self._file_hint) or self._file_hint),
                    "frame_width": int(decode_meta.get("frame_width", 0) or 0),
                    "frame_height": int(decode_meta.get("frame_height", 0) or 0),
                    "decoder_attempts": list(self._decoder_attempts),
                },
            }

    def close(self) -> None:
        try:
            if self._decoder is not None:
                self._decoder.close()
        finally:
            self._decoder = None
            super().close()


class WebcamRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(
        self,
        *,
        device_index: int = 0,
        max_frames: int | None = None,
        source_type: str = "webcam_stream",
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self._cv2 = None
        self._cap = None
        self._device_index = int(device_index)
        self._max_frames = max(1, int(max_frames or 0)) if max_frames else 0
        self._frame_width = max(0, int(frame_width or 0))
        self._frame_height = max(0, int(frame_height or 0))
        self._native_fps = 0.0
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            self._cap = cv2.VideoCapture(self._device_index)
            if not self._cap or not self._cap.isOpened():
                raise RuntimeError("webcam_open_failed")
            if self._frame_width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            if self._frame_height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self._native_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            super().__init__(
                source_kind="webcam",
                source_type=source_type,
                total_items=(self._max_frames or None),
                realtime=True,
            )
        except Exception as exc:
            super().__init__(source_kind="webcam", source_type=source_type, total_items=0, realtime=True)
            self.unavailable = True
            self.last_error = f"webcam_not_available:{exc}"
            self.close()

    def _next_item_impl(self) -> dict[str, Any] | None:
        if self._cap is None or self._cv2 is None:
            return None
        if self._max_frames and self.index >= self._max_frames:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("webcam_capture_failed")
        ok_encode, png = self._cv2.imencode(".png", frame)
        if not ok_encode:
            raise RuntimeError("webcam_png_encode_failed")
        height, width = frame.shape[:2]
        return {
            "image_bytes": bytes(png.tobytes()),
            "source_type": self.source_type,
            "stream_frame_meta": {
                "frame_index": self.index,
                "frame_count_estimate": int(self.total_items or 0),
                "capture_backend": "opencv_webcam",
                "device_index": self._device_index,
                "native_fps": _round4(self._native_fps),
                "frame_width": int(width),
                "frame_height": int(height),
            },
        }

    def close(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        finally:
            self._cap = None
            super().close()


class MicrophoneRealtimeSourceV1(BaseRealtimeSourceV1):
    def __init__(
        self,
        *,
        tick_window_ms: int,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        max_windows: int | None = None,
        source_type: str = "microphone_stream",
    ) -> None:
        self._sounddevice = None
        self._device_index = device_index
        self._sample_rate = max(8000, int(sample_rate or 16000))
        self._channels = max(1, int(channels or 1))
        self._tick_window_ms = max(5, int(tick_window_ms or 50))
        self._frames_per_window = max(1, int(self._sample_rate * (self._tick_window_ms / 1000.0)))
        self._max_windows = max(1, int(max_windows or 0)) if max_windows else 0
        try:
            import sounddevice as sd  # type: ignore

            self._sounddevice = sd
            check_kwargs: dict[str, Any] = {
                "samplerate": self._sample_rate,
                "channels": self._channels,
            }
            if self._device_index is not None:
                check_kwargs["device"] = int(self._device_index)
            sd.check_input_settings(**check_kwargs)
            super().__init__(
                source_kind="microphone",
                source_type=source_type,
                total_items=(self._max_windows or None),
                realtime=True,
            )
        except Exception as exc:
            super().__init__(source_kind="microphone", source_type=source_type, total_items=0, realtime=True)
            self.unavailable = True
            self.last_error = f"microphone_not_available:{exc}"

    def _next_item_impl(self) -> dict[str, Any] | None:
        if self._sounddevice is None:
            return None
        if self._max_windows and self.index >= self._max_windows:
            return None
        kwargs: dict[str, Any] = {
            "samplerate": self._sample_rate,
            "channels": self._channels,
            "dtype": "int16",
            "blocking": True,
        }
        if self._device_index is not None:
            kwargs["device"] = int(self._device_index)
        samples = self._sounddevice.rec(self._frames_per_window, **kwargs)
        audio = np.asarray(samples, dtype=np.int16)
        if audio.ndim == 1:
            audio = audio.reshape((-1, 1))
        pcm = audio.tobytes(order="C")
        payload = BytesIO()
        with wave.open(payload, "wb") as out:
            out.setnchannels(self._channels)
            out.setsampwidth(2)
            out.setframerate(self._sample_rate)
            out.writeframes(pcm)
        peak = int(np.max(np.abs(audio))) if audio.size else 0
        return {
            "audio_bytes": payload.getvalue(),
            "source_type": self.source_type,
            "stream_chunk_meta": {
                "chunk_index": self.index,
                "chunk_count": int(self.total_items or 0),
                "tick_window_ms": self._tick_window_ms,
                "framerate": self._sample_rate,
                "channels": self._channels,
                "sample_width": 2,
                "capture_backend": "sounddevice_rec",
                "device_index": self._device_index if self._device_index is not None else -1,
                "peak_amplitude": peak,
            },
        }


class StreamAdapterV1:
    def split_audio_wav_bytes(
        self,
        raw_bytes: bytes,
        *,
        tick_window_ms: int,
        source_type: str = "audio_stream",
    ) -> list[dict[str, Any]]:
        with wave.open(BytesIO(raw_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        frame_width = max(1, sampwidth * channels)
        frames_per_tick = max(1, int(framerate * (max(5, int(tick_window_ms)) / 1000.0)))
        chunk_size = frames_per_tick * frame_width
        total_chunks = int(math.ceil(len(frames) / max(1, chunk_size)))
        chunks: list[dict[str, Any]] = []
        for index in range(total_chunks):
            start = index * chunk_size
            chunk = frames[start : start + chunk_size]
            if not chunk:
                continue
            payload = BytesIO()
            with wave.open(payload, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(sampwidth)
                out.setframerate(framerate)
                out.writeframes(chunk)
            chunks.append(
                {
                    "audio_bytes": payload.getvalue(),
                    "source_type": source_type,
                    "stream_chunk_meta": {
                        "chunk_index": index,
                        "chunk_count": total_chunks,
                        "tick_window_ms": int(tick_window_ms),
                        "framerate": framerate,
                        "channels": channels,
                        "sample_width": sampwidth,
                        "coverage_ratio": _round4(len(chunk) / max(1, chunk_size)),
                    },
                }
            )
        return chunks

    def split_image_sequence_bytes(
        self,
        frames: list[bytes],
        *,
        source_type: str = "image_stream",
    ) -> list[dict[str, Any]]:
        total = len(frames)
        items: list[dict[str, Any]] = []
        for index, raw in enumerate(frames):
            if not raw:
                continue
            items.append(
                {
                    "image_bytes": raw,
                    "source_type": source_type,
                    "stream_frame_meta": {
                        "frame_index": index,
                        "frame_count": total,
                    },
                }
            )
        return items

    def split_vertical_strip_image(
        self,
        raw_bytes: bytes,
        *,
        frame_count: int,
        source_type: str = "image_stream",
    ) -> list[dict[str, Any]]:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        width, height = image.size
        count = max(1, int(frame_count))
        frame_height = max(1, height // count)
        frames: list[bytes] = []
        for index in range(count):
            upper = index * frame_height
            lower = height if index == count - 1 else min(height, (index + 1) * frame_height)
            frame = image.crop((0, upper, width, lower))
            buf = BytesIO()
            frame.save(buf, format="PNG")
            frames.append(buf.getvalue())
        return self.split_image_sequence_bytes(frames, source_type=source_type)

    def split_video_file_bytes(
        self,
        raw_bytes: bytes,
        *,
        tick_fps: float | None = None,
        frame_stride: int | None = None,
        max_frames: int | None = None,
        source_type: str = "video_stream",
        file_hint: str = "",
    ) -> list[dict[str, Any]]:
        source = self.build_video_file_source(
            raw_bytes=raw_bytes,
            tick_fps=tick_fps,
            frame_stride=frame_stride,
            max_frames=max_frames,
            source_type=source_type,
            file_hint=file_hint,
        )
        if source.unavailable:
            raise RuntimeError(source.last_error or "video stream requires a supported decoder backend")
        items: list[dict[str, Any]] = []
        try:
            while True:
                item = source.next_item()
                if item is None:
                    break
                items.append(dict(item))
        finally:
            source.close()
        return items

    def _guess_video_suffix(self, *, file_hint: str, raw_bytes: bytes) -> str:
        hint = str(file_hint or "").strip().lower()
        if hint.endswith(".mp4"):
            return ".mp4"
        if hint.endswith(".avi"):
            return ".avi"
        if hint.endswith(".mov"):
            return ".mov"
        if hint.endswith(".mkv"):
            return ".mkv"
        if raw_bytes[:4] == b"RIFF":
            return ".avi"
        if raw_bytes[:4] == b"OggS":
            return ".ogv"
        if raw_bytes[:4] == b"\x1aE\xdf\xa3":
            return ".mkv"
        if len(raw_bytes) >= 8 and raw_bytes[4:8] == b"ftyp":
            return ".mp4"
        return ".mp4"

    def merge_stream_items(
        self,
        *,
        texts: list[str] | None = None,
        image_items: list[dict[str, Any]] | None = None,
        audio_items: list[dict[str, Any]] | None = None,
        source_type: str = "multimodal_stream",
    ) -> list[dict[str, Any]]:
        text_rows = [str(item or "") for item in (texts or [])]
        image_rows = list(image_items or [])
        audio_rows = list(audio_items or [])
        total = max(len(text_rows), len(image_rows), len(audio_rows), 1)
        merged: list[dict[str, Any]] = []
        for index in range(total):
            item: dict[str, Any] = {
                "text": text_rows[index] if index < len(text_rows) else "",
                "source_type": source_type,
            }
            if index < len(image_rows):
                item.update(dict(image_rows[index]))
            if index < len(audio_rows):
                item.update(dict(audio_rows[index]))
            merged.append(item)
        return merged

    def build_audio_file_source(
        self,
        *,
        raw_bytes: bytes,
        tick_window_ms: int,
        source_type: str = "audio_stream",
    ) -> BaseRealtimeSourceV1:
        return AudioWavRealtimeSourceV1(
            raw_bytes=raw_bytes,
            tick_window_ms=tick_window_ms,
            source_type=source_type,
        )

    def build_image_sequence_source(
        self,
        *,
        frames: list[bytes] | None = None,
        strip_image_bytes: bytes | None = None,
        frame_count: int = 1,
        source_type: str = "image_stream",
    ) -> BaseRealtimeSourceV1:
        if frames:
            items = self.split_image_sequence_bytes(list(frames), source_type=source_type)
        elif strip_image_bytes is not None:
            items = self.split_vertical_strip_image(strip_image_bytes, frame_count=frame_count, source_type=source_type)
        else:
            items = []
        return SequenceRealtimeSourceV1(source_kind="image_sequence", source_type=source_type, items=items)

    def build_video_file_source(
        self,
        *,
        raw_bytes: bytes,
        tick_fps: float | None = None,
        frame_stride: int | None = None,
        max_frames: int | None = None,
        source_type: str = "video_stream",
        file_hint: str = "",
    ) -> BaseRealtimeSourceV1:
        return OpenCVVideoRealtimeSourceV1(
            raw_bytes=raw_bytes,
            tick_fps=tick_fps,
            frame_stride=frame_stride,
            max_frames=max_frames,
            source_type=source_type,
            file_hint=file_hint,
            suffix_hint=self._guess_video_suffix(file_hint=file_hint, raw_bytes=raw_bytes),
        )

    def build_screen_capture_source(self, *, text_hint: str = "", source_type: str = "screen_capture") -> BaseRealtimeSourceV1:
        return SequenceRealtimeSourceV1(
            source_kind="screen_capture",
            source_type=source_type,
            items=[{"text": str(text_hint or ""), "source_type": source_type, "capture_screen": True}],
            realtime=True,
        )

    def build_webcam_source(
        self,
        *,
        source_type: str = "webcam_stream",
        device_index: int = 0,
        max_frames: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> BaseRealtimeSourceV1:
        return WebcamRealtimeSourceV1(
            source_type=source_type,
            device_index=device_index,
            max_frames=max_frames,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def build_microphone_source(
        self,
        *,
        source_type: str = "microphone_stream",
        tick_window_ms: int = 50,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        max_windows: int | None = None,
    ) -> BaseRealtimeSourceV1:
        return MicrophoneRealtimeSourceV1(
            tick_window_ms=tick_window_ms,
            sample_rate=sample_rate,
            channels=channels,
            device_index=device_index,
            max_windows=max_windows,
            source_type=source_type,
        )


def build_test_wav_bytes(*, sample_rate: int, duration_sec: float, frequency: float, amplitude: int = 12000) -> bytes:
    frames = bytearray()
    sample_count = int(sample_rate * duration_sec)
    for i in range(sample_count):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        frames += struct.pack("<h", sample)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()
