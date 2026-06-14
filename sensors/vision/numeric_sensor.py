from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image
from sensors.reconstruction_payload import make_reconstruction_payload, payload_summary_vector

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _round4(value: float) -> float:
    return round(float(value), 4)


def _resolution_tier(color_shape: list) -> str:
    rows = int(color_shape[0]) if len(color_shape) >= 1 else 0
    if rows >= 20:
        return "focus_high"
    if rows >= 12:
        return "focus_mid"
    if rows > 0:
        return "peripheral_low"
    return "unknown"


def _float_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


def _resize_rgb(image: Image.Image, *, max_side: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    cap = max(32, int(max_side))
    scale = min(1.0, float(cap) / float(max(width, height, 1)))
    if scale >= 1.0:
        return rgb
    return rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _edges(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    if cv2 is not None:
        return (cv2.Canny(gray_u8, 48, 128).astype(np.float32) / 255.0)
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = float(np.percentile(mag, 78.0)) if mag.size else 0.0
    return (mag >= max(0.02, threshold)).astype(np.float32)


def _safe_norm_vector(values: list[float], *, cap: int = 32) -> list[float]:
    clean = [_round4(float(value or 0.0)) for value in values[:cap]]
    return clean


def _downsample_grid(arr: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if arr.size <= 0:
        shape = (height, width) if arr.ndim <= 2 else (height, width, arr.shape[-1])
        return np.zeros(shape, dtype=np.float32)
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (int(width), int(height)), interpolation=cv2.INTER_AREA)
    img_arr = arr
    if arr.ndim == 2:
        image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
        resized = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0
    image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    resized = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_array(arr: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize a numeric payload while preserving its white-box value semantics."""

    return _downsample_grid(arr, width=max(1, int(width)), height=max(1, int(height)))


def _smoothstep(value: float) -> float:
    """Continuous foveal budget curve: smooth enough to avoid hard visual tiers."""

    x = _clamp(float(value or 0.0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _continuous_dim(*, base: int, cap: int, ratio: float) -> int:
    """Interpolate a payload dimension between peripheral budget and focus cap."""

    low = max(1, int(base))
    high = max(low, int(cap))
    smooth = _clamp(float(ratio or 0.0), 0.0, 1.0)
    return max(1, min(high, int(round(low + (high - low) * smooth))))


def _continuous_resolution_tier(ratio: float) -> str:
    """Human-readable label only; cognition uses the continuous values."""

    value = _clamp(float(ratio or 0.0), 0.0, 1.0)
    if value >= 0.88:
        return "near_original_focus"
    if value >= 0.58:
        return "focus_high"
    if value >= 0.28:
        return "focus_soft"
    return "peripheral_low"


def _crop_original_by_resized_bbox(
    original_rgb: np.ndarray | None,
    *,
    resized_width: int,
    resized_height: int,
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray | None:
    """
    Map the object box found in the fixed-budget sensor image back to the input
    image for a small foveal numeric patch.

    This does not store or replay the original image. It only lets the focused
    SA carry a bounded, state-pool numeric detail sample, matching AP's
    humanlike "small fovea, coarse periphery" philosophy.
    """

    if original_rgb is None or original_rgb.size <= 0:
        return None
    orig_h, orig_w = original_rgb.shape[:2]
    if orig_h <= 0 or orig_w <= 0 or resized_width <= 0 or resized_height <= 0:
        return None
    x0 = max(0, min(orig_w - 1, int(math.floor(float(x) / float(resized_width) * float(orig_w)))))
    y0 = max(0, min(orig_h - 1, int(math.floor(float(y) / float(resized_height) * float(orig_h)))))
    x1 = max(x0 + 1, min(orig_w, int(math.ceil(float(x + w) / float(resized_width) * float(orig_w)))))
    y1 = max(y0 + 1, min(orig_h, int(math.ceil(float(y + h) / float(resized_height) * float(orig_h)))))
    crop = original_rgb[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


class NativeVisionNumericSensor:
    """
    Fixed-budget numeric vision sensor.

    It emits global channel SAs and a small number of object-slot SAs. The raw
    image stays outside the state pool; state items carry explicit numeric
    channels for Bn similarity and lightweight reconstruction hints.
    """

    def __init__(self, *, max_objects: int = 4, max_side: int = 160, preview_side: int = 96) -> None:
        self.max_objects = max(1, int(max_objects))
        self.max_side = max(48, int(max_side))
        self.preview_side = max(32, int(preview_side))
        self._previous_gray: np.ndarray | None = None
        self._previous_objects: list[dict] = []

    def ingest_image_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "image_input", focus_state: dict | None = None) -> dict:
        focus = self._normalize_focus_state(focus_state)
        image = Image.open(BytesIO(raw_bytes))
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        resized = _resize_rgb(image, max_side=self.max_side)
        rgb = np.asarray(resized, dtype=np.float32) / 255.0
        gray = _luma(rgb)
        edge = _edges(gray)
        motion_map = self._motion_map(gray)

        global_features = {
            "vision.shape": self._shape_features(edge, mask=None, width=rgb.shape[1], height=rgb.shape[0]),
            "vision.color": self._color_features(rgb, mask=None),
            "vision.spatial": [0.5, 0.5, 1.0, 1.0, _round4(rgb.shape[1] / max(1.0, float(rgb.shape[0]))), 1.0],
            "vision.motion": self._motion_features(motion_map, mask=None),
        }
        field_payloads = self._field_reconstruction_payloads(rgb=rgb, gray=gray, edge=edge, motion_map=motion_map, focus_state=focus)
        objects = self._object_regions(rgb=rgb, edge=edge, motion_map=motion_map, focus_state=focus, original_rgb=original_rgb)
        self._attach_object_motion_vectors(objects)
        state_items = self._build_state_items(
            tick_index=tick_index,
            source_type=source_type,
            global_features=global_features,
            field_payloads=field_payloads,
            objects=objects,
            focus_state=focus,
        )
        inner_vision = self._build_inner_view(
            image=resized,
            global_features=global_features,
            field_payloads=field_payloads,
            objects=objects,
            state_items=state_items,
            focus_state=focus,
        )
        self._previous_gray = gray.copy()
        self._previous_objects = [
            {
                "bbox_norm": list(obj.get("bbox_norm", []) or []),
                "mean_rgb": list(obj.get("mean_rgb", []) or []),
                "object_anchor_id": str(obj.get("object_anchor_id", "") or ""),
            }
            for obj in objects[: self.max_objects]
        ]
        return {
            "packet": {
                "schema_id": "vision_numeric_packet/v1",
                "tick_index": int(tick_index),
                "source_type": source_type,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "object_count": len(objects),
                "focus_state": focus,
            },
            "state_items": state_items,
            "inner_vision": inner_vision,
        }

    def _motion_map(self, gray: np.ndarray) -> np.ndarray:
        if self._previous_gray is None:
            return np.zeros_like(gray, dtype=np.float32)
        prev = self._previous_gray
        if prev.shape != gray.shape:
            if cv2 is not None:
                prev = cv2.resize(prev.astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                prev_img = Image.fromarray(np.clip(prev * 255.0, 0, 255).astype(np.uint8))
                prev = np.asarray(prev_img.resize((gray.shape[1], gray.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return np.abs(gray.astype(np.float32) - prev.astype(np.float32))

    def _object_regions(
        self,
        *,
        rgb: np.ndarray,
        edge: np.ndarray,
        motion_map: np.ndarray,
        focus_state: dict | None = None,
        original_rgb: np.ndarray | None = None,
    ) -> list[dict]:
        focus = dict(focus_state or {})
        height, width = edge.shape
        boxes: list[tuple[int, int, int, int, float]] = []
        edge_u8 = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
        if cv2 is not None:
            contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area_ratio = (w * h) / max(1.0, float(width * height))
                if area_ratio < 0.006:
                    continue
                local_edge = float(edge[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                local_motion = float(motion_map[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                focus_gain = self._focus_gain_for_box(x=x, y=y, w=w, h=h, width=width, height=height, focus_state=focus)
                score = area_ratio * 0.45 + local_edge * 0.42 + min(1.0, local_motion * 4.0) * 0.13 + focus_gain * 0.24
                boxes.append((x, y, w, h, score))
        boxes.extend(self._glyph_like_slice_boxes(edge=edge, motion_map=motion_map, focus_state=focus))
        if not boxes:
            # Deterministic fallback: central window plus four coarse quadrants.
            candidates = [
                (width // 4, height // 4, max(1, width // 2), max(1, height // 2)),
                (0, 0, max(1, width // 2), max(1, height // 2)),
                (width // 2, 0, max(1, width - width // 2), max(1, height // 2)),
                (0, height // 2, max(1, width // 2), max(1, height - height // 2)),
                (width // 2, height // 2, max(1, width - width // 2), max(1, height - height // 2)),
            ]
            for x, y, w, h in candidates:
                local_edge = float(edge[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                local_motion = float(motion_map[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                focus_gain = self._focus_gain_for_box(x=x, y=y, w=w, h=h, width=width, height=height, focus_state=focus)
                score = local_edge * 0.75 + min(1.0, local_motion * 4.0) * 0.25 + focus_gain * 0.24
                boxes.append((x, y, w, h, score))

        boxes.sort(key=lambda row: (-float(row[4]), int(row[0]), int(row[1])))
        selected = []
        for x, y, w, h, score in boxes:
            if len(selected) >= self.max_objects:
                break
            if any(self._iou((x, y, w, h), tuple(prev["bbox_px"])) > 0.62 for prev in selected):
                continue
            crop = rgb[y : y + h, x : x + w]
            original_crop = _crop_original_by_resized_bbox(
                original_rgb,
                resized_width=width,
                resized_height=height,
                x=x,
                y=y,
                w=w,
                h=h,
            )
            edge_crop = edge[y : y + h, x : x + w]
            motion_crop = motion_map[y : y + h, x : x + w]
            mask = self._object_mask(edge_crop)
            spatial = [
                _round4((x + w * 0.5) / max(1.0, float(width))),
                _round4((y + h * 0.5) / max(1.0, float(height))),
                _round4(w / max(1.0, float(width))),
                _round4(h / max(1.0, float(height))),
                _round4(w / max(1.0, float(h))),
                _round4((w * h) / max(1.0, float(width * height))),
            ]
            sampling_focus = self._object_sampling_focus(bbox_norm=spatial[:4], focus_state=focus)
            glyph_slice_like = bool(w <= max(6, width * 0.16) and h >= max(4, height * 0.35))
            object_anchor_id = self._spatial_object_anchor_id(spatial[:4], fine=glyph_slice_like)
            selected.append(
                {
                    "object_anchor_id": object_anchor_id,
                    "bbox_px": [int(x), int(y), int(w), int(h)],
                    "bbox_norm": spatial[:4],
                    "salience": _round4(max(0.08, float(score))),
                    "sampling_focus": sampling_focus,
                    "proposal_kind": "glyph_like_slice" if glyph_slice_like else "edge_object",
                    "numeric_features": {
                        "vision.shape": self._shape_features(edge_crop, mask=mask, width=w, height=h),
                        "vision.color": self._color_features(crop, mask=mask),
                        "vision.spatial": spatial,
                        "vision.motion": self._motion_features(motion_crop, mask=mask),
                    },
                    "mean_rgb": self._mean_rgb(crop),
                    "reconstruction_payloads": self._object_reconstruction_payloads(
                        crop=crop,
                        edge_crop=edge_crop,
                        mask=mask,
                        bbox_norm=spatial[:4],
                        salience=float(score),
                        focus_precision=sampling_focus.get("precision", 0.35),
                        original_crop=original_crop,
                    ),
                }
            )
            selected[-1]["variable_resolution"] = self._resolution_from_payloads(
                selected[-1].get("reconstruction_payloads", {}),
                focus_precision=float((selected[-1].get("sampling_focus", {}) or {}).get("precision", 0.0) or 0.0),
            )
        return selected

    def _attach_object_motion_vectors(self, objects: list[dict]) -> None:
        previous = [row for row in self._previous_objects if isinstance(row, dict)]
        used: set[int] = set()
        for obj in objects:
            bbox = list(obj.get("bbox_norm", []) or [])
            mean_rgb = list(obj.get("mean_rgb", []) or [])
            best_idx = -1
            best_score = 10.0
            for idx, prev in enumerate(previous):
                if idx in used:
                    continue
                prev_bbox = list(prev.get("bbox_norm", []) or [])
                prev_rgb = list(prev.get("mean_rgb", []) or [])
                if len(bbox) < 4 or len(prev_bbox) < 4:
                    continue
                spatial_distance = math.sqrt((float(bbox[0]) - float(prev_bbox[0])) ** 2 + (float(bbox[1]) - float(prev_bbox[1])) ** 2)
                color_distance = 0.0
                if len(mean_rgb) >= 3 and len(prev_rgb) >= 3:
                    color_distance = math.sqrt(sum((float(mean_rgb[i]) - float(prev_rgb[i])) ** 2 for i in range(3))) * 0.18
                score = spatial_distance + color_distance
                if score < best_score:
                    best_score = score
                    best_idx = idx
            dx = dy = speed = 0.0
            continuity = 0.0
            if best_idx >= 0 and best_score <= 0.45:
                used.add(best_idx)
                prev_bbox = list(previous[best_idx].get("bbox_norm", []) or [])
                dx = float(bbox[0]) - float(prev_bbox[0])
                dy = float(bbox[1]) - float(prev_bbox[1])
                speed = math.sqrt(dx * dx + dy * dy)
                continuity = max(0.0, min(1.0, 1.0 - best_score / 0.45))
            vector = [
                _round4(dx),
                _round4(dy),
                _round4(speed),
                _round4(continuity),
                _round4(1.0 if abs(dx) >= abs(dy) and dx < -0.002 else 0.0),
                _round4(1.0 if abs(dx) >= abs(dy) and dx > 0.002 else 0.0),
                _round4(1.0 if abs(dy) > abs(dx) and dy < -0.002 else 0.0),
                _round4(1.0 if abs(dy) > abs(dx) and dy > 0.002 else 0.0),
            ]
            if best_idx >= 0:
                prev_anchor = str(previous[best_idx].get("object_anchor_id", "") or "")
                if prev_anchor:
                    obj["previous_object_anchor_id"] = prev_anchor
            features = dict(obj.get("numeric_features", {}) or {})
            features["vision.motion_vector"] = vector
            obj["numeric_features"] = features
            obj["motion_vector"] = vector

    def _glyph_like_slice_boxes(self, *, edge: np.ndarray, motion_map: np.ndarray, focus_state: dict) -> list[tuple[int, int, int, int, float]]:
        """
        Propose local text-like slices without decoding them.

        These boxes are sensory places AP can foveate and bind through
        teacher-on co-presence. They are not OCR, not sorted answers, and do
        not contain character labels.
        """

        if edge.size <= 0:
            return []
        height, width = edge.shape
        if height < 8 or width / max(1.0, float(height)) < 2.2:
            return []
        column_energy = edge.mean(axis=0)
        if column_energy.size <= 0 or float(column_energy.max()) <= 0.01:
            return []
        threshold = max(0.018, float(np.percentile(column_energy, 58.0)) * 0.72)
        active = column_energy >= threshold
        if active.size >= 3:
            padded = active.copy()
            padded[1:] = np.logical_or(padded[1:], active[:-1])
            padded[:-1] = np.logical_or(padded[:-1], active[1:])
            active = padded

        groups: list[tuple[int, int]] = []
        start = -1
        for index, flag in enumerate(active.tolist() + [False]):
            if flag and start < 0:
                start = index
            elif not flag and start >= 0:
                groups.append((start, index))
                start = -1

        boxes: list[tuple[int, int, int, int, float]] = []
        min_w = max(2, int(round(width * 0.018)))
        max_w = max(min_w + 1, int(round(width * 0.18)))
        for start, end in groups:
            width_px = int(end - start)
            if width_px < min_w:
                continue
            if width_px > max_w:
                chunk_count = max(1, int(round(width_px / max(min_w * 2, width * 0.075))))
                chunk_w = max(min_w, int(round(width_px / chunk_count)))
                subgroups = []
                cursor = start
                while cursor < end:
                    sub_end = min(end, cursor + chunk_w)
                    if sub_end - cursor >= min_w:
                        subgroups.append((cursor, sub_end))
                    cursor = sub_end
            else:
                subgroups = [(start, end)]

            for left, right in subgroups:
                local = edge[:, left:right]
                if local.size <= 0:
                    continue
                row_energy = local.mean(axis=1)
                row_threshold = max(0.012, float(np.percentile(row_energy, 52.0)) * 0.68)
                rows = np.where(row_energy >= row_threshold)[0]
                if rows.size:
                    y1 = max(0, int(rows[0]) - 1)
                    y2 = min(height, int(rows[-1]) + 2)
                else:
                    y1, y2 = 0, height
                x = max(0, int(left) - 1)
                y = max(0, y1)
                w = max(1, min(width - x, int(right - left) + 2))
                h = max(1, min(height - y, int(y2 - y1)))
                area_ratio = (w * h) / max(1.0, float(width * height))
                if area_ratio < 0.004 or area_ratio > 0.28:
                    continue
                local_edge = float(edge[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                local_motion = float(motion_map[y : y + h, x : x + w].mean()) if w > 0 and h > 0 else 0.0
                focus_gain = self._focus_gain_for_box(x=x, y=y, w=w, h=h, width=width, height=height, focus_state=focus_state)
                score = 0.10 + local_edge * 0.58 + area_ratio * 0.18 + min(1.0, local_motion * 4.0) * 0.08 + focus_gain * 0.34
                boxes.append((x, y, w, h, score))
        return boxes[: max(self.max_objects * 4, 12)]

    def _spatial_object_anchor_id(self, bbox_norm: list[float], *, fine: bool = False) -> str:
        """
        Return a stable, white-box spatial handle for a currently seen object.

        Detection slots are frame-local; slot 0 can be the center object in one
        tick and the left object in the next. AP still keeps the slot/order as
        metadata, but the main visual object SA needs a coarse spatial anchor so
        state-pool energy, gaze fatigue, and action-parameter learning do not
        merge unrelated places into one object. This is not class/color
        recognition, only a low-cost visual-space handle that a later online
        tracker can refine.
        """

        x = float((bbox_norm or [0.5])[0] if bbox_norm else 0.5)
        y = float((bbox_norm or [0.5, 0.5])[1] if len(bbox_norm or []) > 1 else 0.5)
        if x < 0.34:
            x_bucket = "left"
        elif x > 0.66:
            x_bucket = "right"
        else:
            x_bucket = "center"
        if y < 0.34:
            y_bucket = "upper"
        elif y > 0.66:
            y_bucket = "lower"
        else:
            y_bucket = "mid"
        if fine:
            fine_x = max(0, min(15, int(x * 16.0)))
            fine_y = max(0, min(7, int(y * 8.0)))
            return f"vision_obj::glyph_slice_x{fine_x:02d}_y{fine_y:02d}"
        return f"vision_obj::{x_bucket}_{y_bucket}"

    def _object_mask(self, edge_crop: np.ndarray) -> np.ndarray:
        if edge_crop.size <= 0:
            return np.zeros((1, 1), dtype=np.float32)
        if cv2 is not None:
            edge_u8 = np.clip(edge_crop * 255.0, 0, 255).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            closed = cv2.dilate(edge_u8, kernel, iterations=1)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(edge_crop.shape, dtype=np.uint8)
            if contours:
                cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
                return (mask.astype(np.float32) / 255.0)
        # Fallback keeps the object region visible even without cv2 contours.
        return np.ones(edge_crop.shape, dtype=np.float32)

    def _palette(self, crop: np.ndarray, *, max_colors: int = 5) -> list[dict]:
        if crop.size <= 0:
            return []
        pixels = np.clip(crop.reshape(-1, 3), 0.0, 1.0)
        quantized = np.round(pixels * 5.0) / 5.0
        colors, counts = np.unique(quantized, axis=0, return_counts=True)
        order = np.argsort(-counts)[: max(1, int(max_colors))]
        total = max(1.0, float(counts.sum()))
        return [
            {
                "rgb": [_round4(value) for value in colors[idx].tolist()],
                "weight": _round4(float(counts[idx]) / total),
            }
            for idx in order
        ]

    def _contour_points(self, mask: np.ndarray, *, max_points: int = 48) -> list[list[float]]:
        if mask.size <= 0:
            return []
        h, w = mask.shape[:2]
        if cv2 is not None:
            mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                epsilon = max(1.0, 0.012 * cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                if len(approx) > max_points:
                    idx = np.linspace(0, len(approx) - 1, max_points).astype(np.int64)
                    approx = approx[idx]
                return [
                    [
                        _round4(float(x) / max(1.0, float(w - 1))),
                        _round4(float(y) / max(1.0, float(h - 1))),
                    ]
                    for x, y in approx.tolist()
                ]
        return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def _field_reconstruction_payloads(self, *, rgb: np.ndarray, gray: np.ndarray, edge: np.ndarray, motion_map: np.ndarray, focus_state: dict | None = None) -> dict[str, dict]:
        color_grid = _downsample_grid(rgb, width=16, height=12)
        luma_grid = _downsample_grid(gray, width=16, height=12)
        edge_grid = _downsample_grid(edge, width=16, height=12)
        motion_grid = _downsample_grid(motion_map, width=16, height=12)
        precision = self._sampling_precision_grid(width=16, height=12, focus_state=focus_state)
        return {
            "vision.field.color_grid": make_reconstruction_payload(
                modality="vision",
                channel="vision.field.color_grid",
                scope="global",
                fidelity_level="low",
                summary_vector=self._grid_summary(color_grid),
                payload_shape=[12, 16, 3],
                payload_values=color_grid.tolist(),
                sampling_precision=0.35,
            ),
            "vision.field.luma_grid": make_reconstruction_payload(
                modality="vision",
                channel="vision.field.luma_grid",
                scope="global",
                fidelity_level="low",
                summary_vector=self._grid_summary(luma_grid),
                payload_shape=[12, 16],
                payload_values=luma_grid.tolist(),
                sampling_precision=0.35,
            ),
            "vision.field.edge_grid": make_reconstruction_payload(
                modality="vision",
                channel="vision.field.edge_grid",
                scope="global",
                fidelity_level="low",
                summary_vector=self._grid_summary(edge_grid),
                payload_shape=[12, 16],
                payload_values=edge_grid.tolist(),
                sampling_precision=0.35,
            ),
            "vision.field.motion_grid": make_reconstruction_payload(
                modality="vision",
                channel="vision.field.motion_grid",
                scope="global",
                fidelity_level="low",
                summary_vector=self._grid_summary(motion_grid),
                payload_shape=[12, 16],
                payload_values=motion_grid.tolist(),
                sampling_precision=0.35,
            ),
            "vision.field.sampling_precision": make_reconstruction_payload(
                modality="vision",
                channel="vision.field.sampling_precision",
                scope="global",
                fidelity_level="low",
                summary_vector=self._grid_summary(precision),
                payload_shape=[12, 16],
                payload_values=precision.tolist(),
                sampling_precision=1.0,
            ),
        }

    def _object_reconstruction_payloads(
        self,
        *,
        crop: np.ndarray,
        original_crop: np.ndarray | None = None,
        edge_crop: np.ndarray,
        mask: np.ndarray,
        bbox_norm: list[float],
        salience: float,
        focus_precision: float = 0.35,
    ) -> dict[str, dict]:
        detail_source = original_crop if original_crop is not None and original_crop.size > 0 else crop
        resolution = self._object_payload_resolution(
            focus_precision=focus_precision,
            crop_shape=crop.shape[:2],
            detail_shape=detail_source.shape[:2],
        )
        mask_grid = _downsample_grid(mask, width=resolution["mask_cols"], height=resolution["mask_rows"])
        color_layout = _downsample_grid(detail_source, width=resolution["color_cols"], height=resolution["color_rows"])
        edge_layout = _downsample_grid(edge_crop, width=resolution["edge_cols"], height=resolution["edge_rows"])
        focus_patch = self._focus_detail_patch(
            crop=detail_source,
            mask=mask,
            focus_precision=focus_precision,
            resolution=resolution,
        )
        palette = self._palette(crop)
        contour = self._contour_points(mask)
        palette_values = []
        for row in palette:
            palette_values.extend(list(row.get("rgb", []) or []) + [float(row.get("weight", 0.0) or 0.0)])
        contour_values = [value for point in contour for value in point]
        precision = max(0.25, min(1.0, float(salience or 0.0) * 4.0, float(focus_precision or 0.0)))
        payloads = {
            "vision.object.mask_grid": make_reconstruction_payload(
                modality="vision",
                channel="vision.object.mask_grid",
                scope="object",
                fidelity_level=resolution["fidelity_level"],
                summary_vector=self._grid_summary(mask_grid) + list(bbox_norm or []),
                payload_shape=[resolution["mask_rows"], resolution["mask_cols"]],
                payload_values=mask_grid.tolist(),
                sampling_precision=precision,
            ),
            "vision.object.palette": make_reconstruction_payload(
                modality="vision",
                channel="vision.object.palette",
                scope="object",
                fidelity_level="mid",
                summary_vector=palette_values[:24],
                payload_shape=[len(palette), 4],
                payload_values=palette_values,
                sampling_precision=precision,
            ),
            "vision.object.contour_points": make_reconstruction_payload(
                modality="vision",
                channel="vision.object.contour_points",
                scope="object",
                fidelity_level="mid",
                summary_vector=contour_values[:32] + list(bbox_norm or []),
                payload_shape=[len(contour), 2],
                payload_values=contour_values,
                sampling_precision=precision,
            ),
            "vision.object.color_layout": make_reconstruction_payload(
                modality="vision",
                channel="vision.object.color_layout",
                scope="object",
                fidelity_level=resolution["fidelity_level"],
                summary_vector=self._grid_summary(color_layout),
                payload_shape=[resolution["color_rows"], resolution["color_cols"], 3],
                payload_values=color_layout.tolist(),
                sampling_precision=precision,
            ),
            "vision.object.edge_layout": make_reconstruction_payload(
                modality="vision",
                channel="vision.object.edge_layout",
                scope="object",
                fidelity_level=resolution["fidelity_level"],
                summary_vector=self._grid_summary(edge_layout),
                payload_shape=[resolution["edge_rows"], resolution["edge_cols"]],
                payload_values=edge_layout.tolist(),
                sampling_precision=precision,
            ),
        }
        if focus_patch.size > 0:
            payloads["vision.object.focus_detail_patch"] = make_reconstruction_payload(
                modality="vision",
                channel="vision.object.focus_detail_patch",
                scope="object_focus",
                fidelity_level="near_original_foveal_patch",
                summary_vector=self._grid_summary(focus_patch),
                payload_shape=[int(focus_patch.shape[0]), int(focus_patch.shape[1]), int(focus_patch.shape[2])],
                payload_values=focus_patch.tolist(),
                sampling_precision=precision,
                payload_limit=12288,
            )
        return payloads

    def _object_payload_resolution(
        self,
        *,
        focus_precision: float,
        crop_shape: tuple[int, int] | None = None,
        detail_shape: tuple[int, int] | None = None,
    ) -> dict:
        """
        Allocate object reconstruction density as a continuous foveated budget.

        AP should not treat every part of the visual field as equally sharp.
        The old three-tier version was useful but too quantized. This uses a
        smooth precision curve so high-focus small objects can approach their
        original crop resolution while peripheral objects remain cheap.
        """

        precision = _clamp(float(focus_precision or 0.0), 0.18, 1.0)
        crop_h = max(1, int((crop_shape or (64, 64))[0] or 1))
        crop_w = max(1, int((crop_shape or (64, 64))[1] or 1))
        detail_h = max(1, int((detail_shape or crop_shape or (64, 64))[0] or 1))
        detail_w = max(1, int((detail_shape or crop_shape or (64, 64))[1] or 1))
        ratio = _smoothstep((precision - 0.18) / 0.82)
        color_rows = _continuous_dim(base=8, cap=min(detail_h, 64), ratio=ratio)
        color_cols = _continuous_dim(base=8, cap=min(detail_w, 64), ratio=ratio)
        mask_rows = _continuous_dim(base=16, cap=min(detail_h, 96), ratio=ratio)
        mask_cols = _continuous_dim(base=16, cap=min(detail_w, 96), ratio=ratio)
        edge_rows = mask_rows
        edge_cols = mask_cols
        tier = _continuous_resolution_tier(ratio)
        return {
            "schema_id": "vision_foveated_payload_resolution/v1",
            "policy": "continuous_foveated_object_payload_resolution",
            "tier": tier,
            "focus_precision": _round4(precision),
            "continuous_ratio": _round4(ratio),
            "crop_shape": [crop_h, crop_w],
            "detail_source_shape": [detail_h, detail_w],
            "color_rows": color_rows,
            "color_cols": color_cols,
            "mask_rows": mask_rows,
            "mask_cols": mask_cols,
            "edge_rows": edge_rows,
            "edge_cols": edge_cols,
            "fidelity_level": f"{tier}_continuous",
        }

    def _focus_detail_patch(self, *, crop: np.ndarray, mask: np.ndarray, focus_precision: float, resolution: dict) -> np.ndarray:
        precision = _clamp(float(focus_precision or 0.0), 0.18, 1.0)
        if precision < 0.66 or crop.size <= 0:
            return np.zeros((0, 0, 3), dtype=np.float32)
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((0, 0, 3), dtype=np.float32)
        if mask.size > 0 and float(mask.max()) > 0.0:
            mask_for_crop = mask
            if mask_for_crop.shape[:2] != crop.shape[:2]:
                mask_for_crop = _resize_array(mask_for_crop, width=w, height=h)
            ys, xs = np.nonzero(mask_for_crop > 0.05)
            cy = int(round(float(ys.mean()))) if ys.size else h // 2
            cx = int(round(float(xs.mean()))) if xs.size else w // 2
        else:
            cy = h // 2
            cx = w // 2
        ratio = float(resolution.get("continuous_ratio", 1.0) or 1.0)
        patch_h = min(h, _continuous_dim(base=12, cap=min(h, 72), ratio=ratio))
        patch_w = min(w, _continuous_dim(base=12, cap=min(w, 72), ratio=ratio))
        y0 = max(0, min(h - patch_h, cy - patch_h // 2))
        x0 = max(0, min(w - patch_w, cx - patch_w // 2))
        patch = crop[y0 : y0 + patch_h, x0 : x0 + patch_w]
        if patch.size <= 0:
            return np.zeros((0, 0, 3), dtype=np.float32)
        # If the patch is still too large for the white-box payload budget,
        # resample it rather than storing an unbounded raw crop.
        max_cells = 4096
        if patch_h * patch_w > max_cells:
            scale = math.sqrt(max_cells / float(patch_h * patch_w))
            target_w = max(1, int(round(patch_w * scale)))
            target_h = max(1, int(round(patch_h * scale)))
            patch = _resize_array(patch, width=target_w, height=target_h)
        return np.asarray(patch, dtype=np.float32)

    def _resolution_from_payloads(self, payloads: dict[str, dict], *, focus_precision: float) -> dict:
        color_shape = list((payloads.get("vision.object.color_layout", {}) or {}).get("payload_shape", []) or [])
        mask_shape = list((payloads.get("vision.object.mask_grid", {}) or {}).get("payload_shape", []) or [])
        edge_shape = list((payloads.get("vision.object.edge_layout", {}) or {}).get("payload_shape", []) or [])
        patch_shape = list((payloads.get("vision.object.focus_detail_patch", {}) or {}).get("payload_shape", []) or [])
        return {
            "schema_id": "vision_foveated_payload_resolution/v1",
            "policy": "continuous_foveated_object_payload_resolution",
            "focus_precision": _round4(float(focus_precision or 0.0)),
            "color_grid_shape": color_shape,
            "mask_grid_shape": mask_shape,
            "edge_grid_shape": edge_shape,
            "focus_detail_patch_shape": patch_shape,
            "near_original_focus_patch": bool(patch_shape),
            "tier": _resolution_tier(color_shape),
        }

    def _grid_summary(self, grid: np.ndarray) -> list[float]:
        arr = np.asarray(grid, dtype=np.float32)
        if arr.size <= 0:
            return [0.0] * 8
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
        mean = flat.mean(axis=0).tolist()
        std = flat.std(axis=0).tolist()
        min_values = flat.min(axis=0).tolist()
        max_values = flat.max(axis=0).tolist()
        return _safe_norm_vector(mean + std + min_values + max_values, cap=24)

    def _sampling_precision_grid(self, *, width: int, height: int, focus_state: dict | None = None) -> np.ndarray:
        focus = self._normalize_focus_state(focus_state)
        ys, xs = np.indices((int(height), int(width)))
        nx = (xs + 0.5) / max(1.0, float(width))
        ny = (ys + 0.5) / max(1.0, float(height))
        cx = _float_default(focus.get("center_x"), 0.5)
        cy = _float_default(focus.get("center_y"), 0.5)
        scale = max(0.35, _float_default(focus.get("scale"), 1.0))
        radius = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        precision = np.clip(1.0 - radius / max(0.18, 0.72 * scale), 0.18, 1.0)
        return precision.astype(np.float32)

    def _shape_features(self, edge: np.ndarray, *, mask: np.ndarray | None, width: int, height: int) -> list[float]:
        if edge.size <= 0:
            return [0.0] * 18
        arr = edge.astype(np.float32)
        active = arr if mask is None else arr * mask.astype(np.float32)
        total = float(active.sum())
        area = max(1.0, float(active.size if mask is None else max(1.0, mask.sum())))
        ys, xs = np.indices(active.shape)
        if total > 1e-9:
            cx = float((xs * active).sum() / total) / max(1.0, float(active.shape[1] - 1))
            cy = float((ys * active).sum() / total) / max(1.0, float(active.shape[0] - 1))
            sx = math.sqrt(float((((xs / max(1.0, active.shape[1] - 1)) - cx) ** 2 * active).sum() / total))
            sy = math.sqrt(float((((ys / max(1.0, active.shape[0] - 1)) - cy) ** 2 * active).sum() / total))
        else:
            cx = cy = 0.5
            sx = sy = 0.0
        radial = self._radial_histogram(active, cx=cx, cy=cy, bins=8)
        hu = self._hu_features(active)
        base = [
            _round4(total / area),
            _round4(cx),
            _round4(cy),
            _round4(sx),
            _round4(sy),
            _round4(width / max(1.0, float(height))),
        ]
        return _safe_norm_vector(base + radial + hu, cap=24)

    def _radial_histogram(self, active: np.ndarray, *, cx: float, cy: float, bins: int) -> list[float]:
        if active.size <= 0:
            return [0.0] * bins
        h, w = active.shape
        ys, xs = np.indices(active.shape)
        nx = xs / max(1.0, float(w - 1))
        ny = ys / max(1.0, float(h - 1))
        radius = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        max_r = max(1e-6, float(radius.max()))
        bucket = np.minimum(bins - 1, np.floor(radius / max_r * bins).astype(np.int32))
        hist = np.zeros((bins,), dtype=np.float32)
        for idx in range(bins):
            hist[idx] = float(active[bucket == idx].sum())
        total = float(hist.sum())
        if total <= 1e-9:
            return [0.0] * bins
        return [_round4(float(value) / total) for value in hist]

    def _hu_features(self, active: np.ndarray) -> list[float]:
        if cv2 is None or active.size <= 0 or float(active.sum()) <= 1e-9:
            return [0.0] * 7
        moments = cv2.moments(np.clip(active * 255.0, 0, 255).astype(np.uint8))
        hu = cv2.HuMoments(moments).reshape(-1).tolist()
        rows = []
        for value in hu[:7]:
            signed = -math.copysign(1.0, float(value)) * math.log10(abs(float(value)) + 1e-12)
            rows.append(_round4(_clamp((signed + 6.0) / 12.0)))
        return rows

    def _color_features(self, rgb: np.ndarray, *, mask: np.ndarray | None) -> list[float]:
        if rgb.size <= 0:
            return [0.0] * 16
        pixels = rgb.reshape(-1, 3)
        if mask is not None and mask.size:
            weights = mask.reshape(-1).astype(np.float32)
            keep = weights > 0
            pixels = pixels[keep] if keep.any() else pixels
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        maxc = pixels.max(axis=1)
        minc = pixels.min(axis=1)
        saturation = np.where(maxc <= 1e-9, 0.0, (maxc - minc) / np.maximum(maxc, 1e-9))
        value = maxc
        hist = np.histogram(value, bins=4, range=(0.0, 1.0))[0].astype(np.float32)
        hist = hist / max(1.0, float(hist.sum()))
        return _safe_norm_vector(
            mean.tolist()
            + std.tolist()
            + [float(saturation.mean()), float(saturation.std()), float(value.mean()), float(value.std())]
            + hist.tolist(),
            cap=16,
        )

    def _motion_features(self, motion_map: np.ndarray, *, mask: np.ndarray | None) -> list[float]:
        if motion_map.size <= 0:
            return [0.0] * 8
        arr = motion_map.astype(np.float32)
        if mask is not None and mask.size == arr.size:
            arr = arr * mask.astype(np.float32)
        total = float(arr.sum())
        ys, xs = np.indices(arr.shape)
        if total > 1e-9:
            cx = float((xs * arr).sum() / total) / max(1.0, float(arr.shape[1] - 1))
            cy = float((ys * arr).sum() / total) / max(1.0, float(arr.shape[0] - 1))
        else:
            cx = cy = 0.5
        return _safe_norm_vector(
            [
                float(arr.mean()),
                float(arr.std()),
                float(arr.max()) if arr.size else 0.0,
                cx,
                cy,
                float((arr > 0.08).mean()),
            ],
            cap=8,
        )

    def _mean_rgb(self, rgb: np.ndarray) -> list[float]:
        if rgb.size <= 0:
            return [0.0, 0.0, 0.0]
        return [_round4(value) for value in rgb.reshape(-1, 3).mean(axis=0).tolist()]

    def _normalize_focus_state(self, focus_state: dict | None) -> dict:
        focus = dict(focus_state or {})
        return {
            "center_x": _round4(_clamp(_float_default(focus.get("center_x"), 0.5), 0.0, 1.0)),
            "center_y": _round4(_clamp(_float_default(focus.get("center_y"), 0.5), 0.0, 1.0)),
            "scale": _round4(_clamp(_float_default(focus.get("scale"), 1.0), 0.35, 1.8)),
            "last_target": str(focus.get("last_target", "") or ""),
            "reconstruction_policy": "foveated_state_pool_numeric_sampling",
        }

    def _focus_gain_for_box(self, *, x: int, y: int, w: int, h: int, width: int, height: int, focus_state: dict) -> float:
        bbox = [
            (float(x) + float(w) * 0.5) / max(1.0, float(width)),
            (float(y) + float(h) * 0.5) / max(1.0, float(height)),
            float(w) / max(1.0, float(width)),
            float(h) / max(1.0, float(height)),
        ]
        return float(self._object_sampling_focus(bbox_norm=bbox, focus_state=focus_state).get("gain", 0.0) or 0.0)

    def _object_sampling_focus(self, *, bbox_norm: list[float], focus_state: dict) -> dict:
        focus = self._normalize_focus_state(focus_state)
        if len(bbox_norm) < 2:
            distance = 1.0
        else:
            dx = float(bbox_norm[0]) - _float_default(focus.get("center_x"), 0.5)
            dy = float(bbox_norm[1]) - _float_default(focus.get("center_y"), 0.5)
            distance = math.sqrt(dx * dx + dy * dy)
        scale = max(0.35, _float_default(focus.get("scale"), 1.0))
        focus_radius = max(0.12, 0.42 * scale)
        gain = _clamp(1.0 - distance / focus_radius, 0.0, 1.0)
        precision = _clamp(0.24 + gain * 0.76, 0.18, 1.0)
        return {
            "schema_id": "vision_foveated_sampling_focus/v1",
            "distance": _round4(distance),
            "gain": _round4(gain),
            "precision": _round4(precision),
            "focus_radius": _round4(focus_radius),
            "center_x": focus["center_x"],
            "center_y": focus["center_y"],
            "scale": focus["scale"],
        }

    def _build_state_items(
        self,
        *,
        tick_index: int,
        source_type: str,
        global_features: dict[str, list[float]],
        field_payloads: dict[str, dict],
        objects: list[dict],
        focus_state: dict | None = None,
    ) -> list[dict]:
        focus = self._normalize_focus_state(focus_state)
        items: list[dict] = []
        for channel in ("vision.shape", "vision.color", "vision.spatial", "vision.motion"):
            short = channel.split(".")[-1]
            energy = 0.55 if short != "motion" else max(0.08, min(0.75, global_features[channel][0] * 4.0 if global_features[channel] else 0.08))
            items.append(
                {
                    "sa_label": f"vision::global::{short}",
                    "display_text": f"vision global {short}",
                    "source_type": "vision_numeric",
                    "family": "vision_channel",
                    "position": len(items),
                    "real_energy": _round4(energy),
                    "numeric_features": {channel: global_features[channel]},
                    "anchor_meta": {
                        "channel": channel,
                        "tick_index": int(tick_index),
                        "source_type": source_type,
                        "feature_scope": "global",
                        "sensor_focus_state": focus,
                    },
                }
            )
        for channel, payload in sorted((field_payloads or {}).items()):
            short = channel.split(".")[-1]
            items.append(
                {
                    "sa_label": f"vision::field::{short}",
                    "display_text": f"vision field {short}",
                    "source_type": "vision_numeric",
                    "family": "vision_channel",
                    "position": len(items),
                    "real_energy": _round4(0.36 if short != "motion_grid" else 0.18),
                    "numeric_features": {channel: payload_summary_vector(payload)},
                    "reconstruction_payload": payload,
                    "anchor_meta": {
                        "channel": channel,
                        "tick_index": int(tick_index),
                        "source_type": source_type,
                        "feature_scope": "global_reconstruction_payload",
                        "sensor_focus_state": focus,
                    },
                }
            )
        for idx, obj in enumerate(objects[: self.max_objects]):
            object_anchor_id = str(obj.get("object_anchor_id", "") or f"vision_obj::slot_{idx}")
            features = dict(obj.get("numeric_features", {}) or {})
            sampling_focus = dict(obj.get("sampling_focus", {}) or {})
            features["vision.focus"] = [
                float(sampling_focus.get("precision", 0.0) or 0.0),
                float(sampling_focus.get("distance", 1.0) or 1.0),
                _float_default(focus.get("center_x"), 0.5),
                _float_default(focus.get("center_y"), 0.5),
                _float_default(focus.get("scale"), 1.0),
            ]
            payloads = dict(obj.get("reconstruction_payloads", {}) or {})
            for channel, payload in payloads.items():
                features[channel] = payload_summary_vector(payload)
            salience = _round4(max(0.08, min(1.4, float(obj.get("salience", 0.0) or 0.0) * 3.0)))
            items.append(
                {
                    "sa_label": object_anchor_id,
                    "display_text": f"vision object {object_anchor_id.removeprefix('vision_obj::')}",
                    "source_type": "vision_numeric",
                    "family": "vision_object",
                    "position": len(items),
                    "real_energy": salience,
                    "numeric_features": features,
                    "reconstruction_payload": {
                        "schema_id": "reconstruction_payload_bundle/v1",
                        "modality": "vision",
                        "scope": "object",
                        "channels": payloads,
                    },
                    "anchor_meta": {
                        "channel": "vision.object",
                        "tick_index": int(tick_index),
                        "track_slot": idx,
                        "object_anchor_id": object_anchor_id,
                        "legacy_slot_label": f"vision_obj::slot_{idx}",
                        "previous_object_anchor_id": str(obj.get("previous_object_anchor_id", "") or ""),
                        "proposal_kind": str(obj.get("proposal_kind", "") or "edge_object"),
                        "feature_channels": sorted(features),
                        "bbox_norm": list(obj.get("bbox_norm", []) or []),
                        "mean_rgb": list(obj.get("mean_rgb", []) or []),
                        "reconstruction_channels": sorted(payloads),
                        "sampling_focus": sampling_focus,
                        "variable_resolution": dict(obj.get("variable_resolution", {}) or {}),
                        "sensor_focus_state": focus,
                        "learnable_handle": True,
                    },
                }
            )
            for channel in ("vision.shape", "vision.color", "vision.spatial", "vision.motion", "vision.motion_vector"):
                short = channel.split(".")[-1]
                items.append(
                    {
                        "sa_label": f"vision::obj::{idx}::{short}",
                        "display_text": f"vision object {idx} {short}",
                        "source_type": "vision_numeric",
                        "family": "vision_channel",
                        "position": len(items),
                        "real_energy": _round4(max(0.04, salience * 0.55)),
                        "numeric_features": {channel: features.get(channel, [])},
                        "anchor_meta": {
                            "channel": channel,
                            "tick_index": int(tick_index),
                            "track_slot": idx,
                            "object_anchor_id": object_anchor_id,
                            "parent_object_label": object_anchor_id,
                            "bbox_norm": list(obj.get("bbox_norm", []) or []),
                            "sampling_focus": sampling_focus,
                            "variable_resolution": dict(obj.get("variable_resolution", {}) or {}),
                            "sensor_focus_state": focus,
                        },
                    }
                )
        return items

    def _build_inner_view(
        self,
        *,
        image: Image.Image,
        global_features: dict,
        field_payloads: dict[str, dict],
        objects: list[dict],
        state_items: list[dict],
        focus_state: dict | None = None,
    ) -> dict:
        focus = self._normalize_focus_state(focus_state)
        return {
            "schema_id": "inner_vision_numeric/v1",
            "current_frame": {
                "width": int(image.size[0]),
                "height": int(image.size[1]),
                "reconstruction_basis": "state_pool_numeric_channels",
                "raw_preview_payload": False,
                "sensor_focus_state": focus,
            },
            "global_channels": sorted(global_features),
            "field_reconstruction": {
                "schema_id": "vision_field_reconstruction/v1",
                "reconstruction_basis": "state_pool_numeric_channels",
                "payloads": dict(field_payloads or {}),
            },
            "focus_objects": [
                str(obj.get("object_anchor_id", "") or f"vision_obj::slot_{idx}")
                for idx, obj in enumerate(objects)
            ],
            "object_reconstruction": [
                {
                    "slot": idx,
                    "object_anchor_id": str(obj.get("object_anchor_id", "") or f"vision_obj::slot_{idx}"),
                    "legacy_slot_label": f"vision_obj::slot_{idx}",
                    "bbox_norm": list(obj.get("bbox_norm", []) or []),
                    "mean_rgb": list(obj.get("mean_rgb", []) or []),
                    "motion_vector": list(obj.get("motion_vector", []) or []),
                    "salience": _round4(float(obj.get("salience", 0.0) or 0.0)),
                    "sampling_focus": dict(obj.get("sampling_focus", {}) or {}),
                    "variable_resolution": dict(obj.get("variable_resolution", {}) or {}),
                    "reconstruction_payload": {
                        "schema_id": "reconstruction_payload_bundle/v1",
                        "modality": "vision",
                        "scope": "object",
                        "channels": dict(obj.get("reconstruction_payloads", {}) or {}),
                    },
                }
                for idx, obj in enumerate(objects)
            ],
            "energy_summary": {
                "state_item_count": len(state_items),
                "object_count": len(objects),
            },
            "recall_layers": [],
            "prediction_layers": [],
        }

    def _iou(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        x1 = max(lx, rx)
        y1 = max(ly, ry)
        x2 = min(lx + lw, rx + rw)
        y2 = min(ly + lh, ry + rh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = max(1, lw * lh + rw * rh - inter)
        return float(inter) / float(union)
