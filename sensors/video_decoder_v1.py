# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _round4(value: float) -> float:
    return round(float(value), 4)


def _import_av_module():
    import av  # type: ignore

    return av


def _import_cv2_module():
    import cv2  # type: ignore

    return cv2


class BaseVideoDecoderV1:
    def __init__(
        self,
        *,
        backend_name: str,
        file_hint: str,
        suffix_hint: str,
        uses_tempfile: bool,
    ) -> None:
        self.backend_name = str(backend_name or "unknown_video_decoder")
        self.file_hint = str(file_hint or "")
        self.suffix_hint = str(suffix_hint or ".mp4")
        self.uses_tempfile = bool(uses_tempfile)
        self.total_frames = 0
        self.native_fps = 25.0
        self._closed = False

    def read_frame_png(self) -> tuple[int, bytes, dict[str, Any]] | None:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "file_hint": self.file_hint,
            "suffix_hint": self.suffix_hint,
            "uses_tempfile": self.uses_tempfile,
            "total_frames": int(self.total_frames or 0),
            "native_fps": _round4(self.native_fps),
            "closed": self._closed,
        }


class PyAVBytesVideoDecoderV1(BaseVideoDecoderV1):
    def __init__(self, *, raw_bytes: bytes, file_hint: str = "", suffix_hint: str = ".mp4") -> None:
        super().__init__(
            backend_name="pyav_bytes_memory",
            file_hint=file_hint,
            suffix_hint=suffix_hint,
            uses_tempfile=False,
        )
        self._buffer = BytesIO(raw_bytes)
        self._container = None
        self._stream = None
        self._decoded_index = 0
        try:
            av = _import_av_module()
            self._container = av.open(self._buffer, mode="r")
            video_streams = [stream for stream in list(self._container.streams) if str(getattr(stream, "type", "")) == "video"]
            if not video_streams:
                raise RuntimeError("video_stream_not_found")
            self._stream = video_streams[0]
            average_rate = getattr(self._stream, "average_rate", None)
            if average_rate:
                self.native_fps = float(average_rate)
            elif float(getattr(self._stream, "base_rate", 0.0) or 0.0) > 0.0:
                self.native_fps = float(getattr(self._stream, "base_rate", 0.0) or 0.0)
            if self.native_fps <= 0.0:
                self.native_fps = 25.0
            self.total_frames = int(getattr(self._stream, "frames", 0) or 0)
            self._frame_iter = iter(self._container.decode(video=0))
        except Exception:
            self.close()
            raise

    def read_frame_png(self) -> tuple[int, bytes, dict[str, Any]] | None:
        if self._container is None:
            return None
        try:
            frame = next(self._frame_iter)
        except StopIteration:
            return None
        rgb = frame.to_ndarray(format="rgb24")
        if not isinstance(rgb, np.ndarray):
            rgb = np.asarray(rgb, dtype=np.uint8)
        image = Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        decoded_index = self._decoded_index
        self._decoded_index += 1
        width = int(getattr(frame, "width", image.width) or image.width)
        height = int(getattr(frame, "height", image.height) or image.height)
        return (
            decoded_index,
            buf.getvalue(),
            {
                "decode_backend": self.backend_name,
                "decode_mode": "memory_bytes",
                "frame_width": width,
                "frame_height": height,
                "temp_suffix": "",
                "file_hint": self.file_hint,
            },
        )

    def close(self) -> None:
        try:
            if self._container is not None:
                self._container.close()
        finally:
            self._container = None
            self._stream = None
            self._buffer.close()
            super().close()


class OpenCVTempfileVideoDecoderV1(BaseVideoDecoderV1):
    def __init__(self, *, raw_bytes: bytes, file_hint: str = "", suffix_hint: str = ".mp4") -> None:
        super().__init__(
            backend_name="opencv_videocapture_lazy",
            file_hint=file_hint,
            suffix_hint=suffix_hint,
            uses_tempfile=True,
        )
        self._cv2 = None
        self._cap = None
        self._tmp_path: Path | None = None
        self._decoded_index = 0
        try:
            cv2 = _import_cv2_module()
            self._cv2 = cv2
            with tempfile.NamedTemporaryFile(delete=False, suffix=self.suffix_hint) as tmp:
                tmp.write(raw_bytes)
                tmp.flush()
                self._tmp_path = Path(tmp.name)
            self._cap = cv2.VideoCapture(str(self._tmp_path))
            if not self._cap or not self._cap.isOpened():
                raise RuntimeError("无法打开视频文件，可能缺少解码器或文件损坏。")
            self.native_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.native_fps <= 0.0:
                self.native_fps = 25.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            self.close()
            raise

    def read_frame_png(self) -> tuple[int, bytes, dict[str, Any]] | None:
        if self._cap is None or self._cv2 is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        ok_encode, png = self._cv2.imencode(".png", frame)
        if not ok_encode:
            raise RuntimeError("opencv_png_encode_failed")
        decoded_index = self._decoded_index
        self._decoded_index += 1
        height = int(frame.shape[0]) if hasattr(frame, "shape") and len(frame.shape) >= 1 else 0
        width = int(frame.shape[1]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else 0
        return (
            decoded_index,
            bytes(png.tobytes()),
            {
                "decode_backend": self.backend_name,
                "decode_mode": "tempfile_path",
                "frame_width": width,
                "frame_height": height,
                "temp_suffix": self.suffix_hint,
                "file_hint": self.file_hint,
            },
        )

    def close(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        finally:
            self._cap = None
            if self._tmp_path is not None:
                try:
                    self._tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._tmp_path = None
            super().close()


def open_video_decoder_v1(
    *,
    raw_bytes: bytes,
    file_hint: str = "",
    suffix_hint: str = ".mp4",
) -> tuple[BaseVideoDecoderV1, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None
    builders = [
        ("pyav_bytes_memory", PyAVBytesVideoDecoderV1),
        ("opencv_videocapture_lazy", OpenCVTempfileVideoDecoderV1),
    ]
    for backend_name, builder in builders:
        try:
            decoder = builder(raw_bytes=raw_bytes, file_hint=file_hint, suffix_hint=suffix_hint)
            attempts.append(
                {
                    "backend": backend_name,
                    "ok": True,
                    "uses_tempfile": bool(decoder.uses_tempfile),
                }
            )
            return decoder, attempts
        except Exception as exc:
            last_exc = exc
            attempts.append(
                {
                    "backend": backend_name,
                    "ok": False,
                    "uses_tempfile": bool(backend_name == "opencv_videocapture_lazy"),
                    "error": str(exc),
                }
            )
    if last_exc is not None:
        raise RuntimeError(
            "video_decoder_open_failed: "
            + " | ".join(
                f"{item.get('backend')}={item.get('error', 'ok')}"
                for item in attempts
            )
        ) from last_exc
    raise RuntimeError("video_decoder_open_failed:no_backend_attempted")
