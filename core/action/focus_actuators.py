from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class VisualGazeActuator:
    """
    Internal visual focus controller.

    It does not move a real camera or replay source assets. It keeps AP's
    current gaze center / sampling scale as first-class state so later sensory
    sampling can be biased by the same action loop that produced the intent.
    """

    def __init__(self) -> None:
        self.center_x = 0.5
        self.center_y = 0.5
        self.scale = 1.0
        self.last_target = ""

    def step(self, *, tick_index: int, selected_actions: list[dict], attention_trace: dict) -> dict:
        items: list[dict] = []
        events: list[dict] = []
        selected_labels = [str(label or "") for label in list((attention_trace or {}).get("selected_labels", []) or []) if str(label or "")]
        for row in selected_actions or []:
            action_id = str((row or {}).get("action_id", "") or "")
            if action_id not in {
                "action::move_gaze_to",
                "action::nudge_gaze",
                "action::scan_visual_field",
                "action::hold_gaze",
                "action::zoom_visual_focus",
                "action::widen_visual_focus",
            }:
                continue
            decisiveness = _clamp(float((row or {}).get("effective_decisiveness", (row or {}).get("drive", 0.0)) or 0.0), 0.0, 1.0)
            params = dict((row or {}).get("params", {}) or {})
            event = self._apply_action(action_id=action_id, params=params, selected_labels=selected_labels, decisiveness=decisiveness)
            event.update(
                {
                    "schema_id": "visual_gaze_control_event/v1",
                    "tick_index": int(tick_index),
                    "action_id": action_id,
                    "drive": _round4(float((row or {}).get("drive", 0.0) or 0.0)),
                    "decisiveness": _round4(decisiveness),
                }
            )
            events.append(event)
            items.append(self._item_from_event(event))
        return {
            "schema_id": "visual_gaze_actuator_trace/v1",
            "state": self.state(),
            "events": events,
            "items": items,
        }

    def state(self) -> dict:
        return {
            "center_x": _round4(self.center_x),
            "center_y": _round4(self.center_y),
            "scale": _round4(self.scale),
            "last_target": self.last_target,
        }

    def _apply_action(self, *, action_id: str, params: dict, selected_labels: list[str], decisiveness: float) -> dict:
        target = str(params.get("gaze_target_key", "") or params.get("target", "") or (selected_labels[0] if selected_labels else ""))
        old_center_x = self.center_x
        old_center_y = self.center_y
        old_scale = self.scale
        if action_id == "action::move_gaze_to":
            if not self._has_absolute_visual_target(params):
                return {
                    "control_kind": "move_gaze_to_skipped",
                    "target": target,
                    "center_x": _round4(self.center_x),
                    "center_y": _round4(self.center_y),
                    "scale": _round4(self.scale),
                    "old_center_x": _round4(old_center_x),
                    "old_center_y": _round4(old_center_y),
                    "old_scale": _round4(old_scale),
                    "movement_distance": 0.0,
                    "skipped": True,
                    "reason": "missing_visual_target_params",
                    "parameter_requirement": "move_gaze_to_requires_x_y_or_bbox_norm",
                }
            x, y = self._target_xy(params=params, selected_labels=selected_labels)
            self.center_x = x
            self.center_y = y
            self.last_target = target
            control_kind = "move_gaze_to"
        elif action_id == "action::nudge_gaze":
            dx = float(params.get("dx", 0.08) or 0.08)
            dy = float(params.get("dy", 0.0) or 0.0)
            self.center_x = _clamp(self.center_x + dx * max(0.35, decisiveness), 0.0, 1.0)
            self.center_y = _clamp(self.center_y + dy * max(0.35, decisiveness), 0.0, 1.0)
            control_kind = "nudge_gaze"
        elif action_id == "action::scan_visual_field":
            # Deterministic bounded scan pattern; no hidden randomness.
            self.center_x = _clamp(1.0 - self.center_x, 0.0, 1.0)
            self.center_y = _clamp(0.25 if self.center_y > 0.5 else 0.75, 0.0, 1.0)
            control_kind = "scan_visual_field"
        elif action_id == "action::hold_gaze":
            self.last_target = target or self.last_target
            control_kind = "hold_gaze"
        elif action_id == "action::zoom_visual_focus":
            scale = float(params.get("scale", 0.72) or 0.72)
            self.scale = _clamp(min(self.scale, scale) - decisiveness * 0.08, 0.35, 1.8)
            self.last_target = target or self.last_target
            control_kind = "zoom_visual_focus"
        else:
            scale = float(params.get("scale", 1.25) or 1.25)
            self.scale = _clamp(max(self.scale, scale) + decisiveness * 0.08, 0.35, 1.8)
            control_kind = "widen_visual_focus"
        distance = ((self.center_x - old_center_x) ** 2 + (self.center_y - old_center_y) ** 2) ** 0.5
        event = {
            "control_kind": control_kind,
            "target": target,
            "center_x": _round4(self.center_x),
            "center_y": _round4(self.center_y),
            "scale": _round4(self.scale),
            "old_center_x": _round4(old_center_x),
            "old_center_y": _round4(old_center_y),
            "old_scale": _round4(old_scale),
            "movement_distance": _round4(distance),
        }
        if params.get("bbox_norm"):
            event["bbox_norm"] = list(params.get("bbox_norm", []) or [])[:4]
        if params.get("reason"):
            event["reason"] = str(params.get("reason", "") or "")
        if params.get("target") and str(params.get("target", "") or "") != target:
            event["source_target"] = str(params.get("target", "") or "")
        if params.get("gaze_target_key"):
            event["gaze_target_key"] = str(params.get("gaze_target_key", "") or "")
        if params.get("score_components"):
            event["score_components"] = dict(params.get("score_components", {}) or {})
        return event

    def _has_absolute_visual_target(self, params: dict) -> bool:
        """
        `move_gaze_to` is a parameterized visual action, not "look at whatever
        label is currently salient". Requiring explicit x/y or bbox keeps
        learned gaze experience about real movement parameters and prevents a
        text/internal label from being hashed into a fake eye movement.
        """

        if "x" in params or "y" in params:
            return True
        bbox = list(params.get("bbox_norm", []) or [])
        return len(bbox) >= 2

    def _target_xy(self, *, params: dict, selected_labels: list[str]) -> tuple[float, float]:
        if "x" in params or "y" in params:
            return (
                _clamp(float(params.get("x", self.center_x) or self.center_x), 0.0, 1.0),
                _clamp(float(params.get("y", self.center_y) or self.center_y), 0.0, 1.0),
            )
        bbox = list(params.get("bbox_norm", []) or [])
        if len(bbox) >= 2:
            # Real visual-object actions should land on the object's normalized
            # center. The old label hash remains only as a non-visual fallback.
            return (
                _clamp(float(bbox[0] or self.center_x), 0.0, 1.0),
                _clamp(float(bbox[1] or self.center_y), 0.0, 1.0),
            )
        return self.center_x, self.center_y

    def _item_from_event(self, event: dict) -> dict:
        return {
            "sa_label": "control::visual_gaze",
            "display_text": "视觉焦点控制",
            "family": "action_control",
            "source_type": "action_control",
            "real_energy": _round4(0.12 + float(event.get("decisiveness", 0.0) or 0.0) * 0.24),
            "virtual_energy": _round4(0.18 + (1.0 - float(event.get("scale", 1.0) or 1.0)) * 0.08),
            "anchor_meta": dict(event),
            "numeric_features": {
                "visual_gaze": [float(event.get("center_x", 0.5) or 0.5), float(event.get("center_y", 0.5) or 0.5), float(event.get("scale", 1.0) or 1.0)],
            },
        }


class AuditoryBandActuator:
    """
    Internal auditory focus controller.

    It updates AP's listened frequency window. The audio reconstruction path can
    later use this state to make focus-band STFT/envelope sampling sharper near
    the attended band and blurrier away from it.
    """

    def __init__(self) -> None:
        self.center_hz = 1000.0
        self.width_hz = 2400.0
        self.last_target = ""

    def step(self, *, tick_index: int, selected_actions: list[dict], attention_trace: dict) -> dict:
        items: list[dict] = []
        events: list[dict] = []
        selected_labels = [str(label or "") for label in list((attention_trace or {}).get("selected_labels", []) or []) if str(label or "")]
        for row in selected_actions or []:
            action_id = str((row or {}).get("action_id", "") or "")
            if action_id not in {
                "action::slide_audio_band",
                "action::lock_audio_band",
                "action::narrow_audio_band",
                "action::widen_audio_band",
            }:
                continue
            decisiveness = _clamp(float((row or {}).get("effective_decisiveness", (row or {}).get("drive", 0.0)) or 0.0), 0.0, 1.0)
            params = dict((row or {}).get("params", {}) or {})
            event = self._apply_action(action_id=action_id, params=params, selected_labels=selected_labels, decisiveness=decisiveness)
            event.update(
                {
                    "schema_id": "auditory_band_control_event/v1",
                    "tick_index": int(tick_index),
                    "action_id": action_id,
                    "drive": _round4(float((row or {}).get("drive", 0.0) or 0.0)),
                    "decisiveness": _round4(decisiveness),
                }
            )
            events.append(event)
            items.append(self._item_from_event(event))
        return {
            "schema_id": "auditory_band_actuator_trace/v1",
            "state": self.state(),
            "events": events,
            "items": items,
        }

    def state(self) -> dict:
        return {
            "center_hz": _round4(self.center_hz),
            "width_hz": _round4(self.width_hz),
            "last_target": self.last_target,
        }

    def _apply_action(self, *, action_id: str, params: dict, selected_labels: list[str], decisiveness: float) -> dict:
        target = str(params.get("target", "") or (selected_labels[0] if selected_labels else ""))
        if action_id in {"action::slide_audio_band", "action::lock_audio_band"}:
            center = params.get("center_hz")
            if center is None:
                seed = sum(ord(ch) for ch in (target or "audio"))
                center = 160.0 + float(seed % 5600)
            target_center = _clamp(float(center or self.center_hz), 40.0, 8000.0)
            if action_id == "action::lock_audio_band":
                self.center_hz = target_center
                control_kind = "lock_audio_band"
            else:
                mix = max(0.28, decisiveness)
                self.center_hz = _clamp(self.center_hz * (1.0 - mix) + target_center * mix, 40.0, 8000.0)
                control_kind = "slide_audio_band"
            self.last_target = target
        elif action_id == "action::narrow_audio_band":
            width = float(params.get("width_hz", 900.0) or 900.0)
            self.width_hz = _clamp(min(self.width_hz, width) - decisiveness * 80.0, 120.0, 8000.0)
            control_kind = "narrow_audio_band"
        else:
            width = float(params.get("width_hz", 3600.0) or 3600.0)
            self.width_hz = _clamp(max(self.width_hz, width) + decisiveness * 120.0, 120.0, 8000.0)
            control_kind = "widen_audio_band"
        return {
            "control_kind": control_kind,
            "target": target,
            "center_hz": _round4(self.center_hz),
            "width_hz": _round4(self.width_hz),
        }

    def _item_from_event(self, event: dict) -> dict:
        return {
            "sa_label": "control::auditory_band",
            "display_text": "听觉焦段控制",
            "family": "action_control",
            "source_type": "action_control",
            "real_energy": _round4(0.12 + float(event.get("decisiveness", 0.0) or 0.0) * 0.22),
            "virtual_energy": _round4(0.12 + max(0.0, 1800.0 - float(event.get("width_hz", 2400.0) or 2400.0)) / 12000.0),
            "anchor_meta": dict(event),
            "numeric_features": {
                "auditory_band": [float(event.get("center_hz", 1000.0) or 1000.0), float(event.get("width_hz", 2400.0) or 2400.0)],
            },
        }
