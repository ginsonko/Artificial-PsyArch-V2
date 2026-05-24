# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageGrab

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class ExecutorConfig:
    enabled: bool
    dry_run: bool
    max_actions_per_tick: int
    screenshot_enabled: bool
    screenshot_scale: float
    type_interval_ms: int


class ComputerExecutorV2:
    def __init__(self, *, config: ExecutorConfig, max_events: int = 256) -> None:
        self.config = config
        self.max_events = max(16, int(max_events))
        self._events: list[dict[str, Any]] = []
        self._allowed_actions = {
            "move_mouse",
            "click",
            "double_click",
            "scroll",
            "type_text",
            "press_key",
            "move_gaze",
            "continue_focus",
            "inspect_residual",
            "move_audio_focus",
            "continue_audio_focus",
            "inspect_audio_residual",
            "wait",
            "noop",
        }

    def evaluate_and_execute(
        self,
        *,
        tick_index: int,
        action_drives: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected = []
        for item in action_drives:
            if not isinstance(item, dict):
                continue
            if float(item.get("firmness", 0.0) or 0.0) <= 0.0:
                continue
            action_id = str(item.get("action_id", "") or "")
            action_name = action_id.replace("action::", "")
            if action_name not in self._allowed_actions:
                continue
            selected.append(
                {
                    "action_id": action_id,
                    "action_name": action_name,
                    "drive": float(item.get("drive", 0.0) or 0.0),
                    "actuator_id": str(item.get("actuator_id", "") or ""),
                    "instance_id": str(item.get("instance_id", "") or ""),
                    "firmness": float(item.get("firmness", 0.0) or 0.0),
                    "firmness_norm": float(item.get("firmness_norm", 0.0) or 0.0),
                    "reason": str(item.get("reason", "") or ""),
                    "params": dict(item.get("params", {}) or {}),
                }
            )
        selected.sort(key=lambda row: (-float(row.get("drive", 0.0) or 0.0), row.get("action_id", "")))
        chosen = selected[: max(1, int(self.config.max_actions_per_tick))]
        executed: list[dict[str, Any]] = []
        for row in chosen:
            result = self._execute_single(row)
            event = {"tick_index": int(tick_index), **row, **result}
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]
            executed.append(event)
        return {
            "selected_actions": executed,
            "recent_events": list(self._events)[-8:],
            "allowed_actions": sorted(self._allowed_actions),
            "executor_status": self.status(),
        }

    def capture_screenshot_packet(self, *, force: bool = False) -> dict[str, Any]:
        if not self.config.screenshot_enabled and not force:
            return {"enabled": False, "captured": False}
        try:
            image = ImageGrab.grab()
            width, height = image.size
            scale = _clamp(self.config.screenshot_scale, 0.05, 1.0)
            preview_image = image.copy()
            preview_image.thumbnail((640, 360), Image.BILINEAR)
            preview_buffer = io.BytesIO()
            preview_image.save(preview_buffer, format="PNG")
            preview_png = preview_buffer.getvalue()
            if scale < 0.999:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            return {
                "enabled": True,
                "captured": True,
                "image_bytes": raw,
                "meta": {
                    "original_size": {"width": width, "height": height},
                    "scaled_size": {"width": image.size[0], "height": image.size[1]},
                    "scale": scale,
                    "preview_size": {"width": preview_image.size[0], "height": preview_image.size[1]},
                    "captured_at_ms": int(time.time() * 1000),
                },
                "preview_b64": base64.b64encode(preview_png).decode("ascii"),
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            return {
                "enabled": True,
                "captured": False,
                "error": str(exc),
                "meta": {
                    "captured_at_ms": int(time.time() * 1000),
                    "scale": _clamp(self.config.screenshot_scale, 0.05, 1.0),
                },
            }

    def export_payload(self) -> dict[str, Any]:
        return {
            "events": list(self._events),
            "allowed_actions": sorted(self._allowed_actions),
            "config": {
                "enabled": self.config.enabled,
                "dry_run": self.config.dry_run,
                "max_actions_per_tick": self.config.max_actions_per_tick,
                "screenshot_enabled": self.config.screenshot_enabled,
                "screenshot_scale": self.config.screenshot_scale,
                "type_interval_ms": self.config.type_interval_ms,
            },
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._events = list(payload.get("events", []) or [])[-self.max_events :]
        allowed = payload.get("allowed_actions", []) or []
        if allowed:
            self._allowed_actions = {str(item or "") for item in allowed if str(item or "")}

    def status(self) -> dict[str, Any]:
        last = self._events[-1] if self._events else {}
        return {
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "screenshot_enabled": self.config.screenshot_enabled,
            "backend": "pyautogui" if pyautogui is not None else "none",
            "allowed_actions": sorted(self._allowed_actions),
            "event_count": len(self._events),
            "last_action": str(last.get("action_name", "") or ""),
            "last_status": str(last.get("status", "") or ""),
        }

    def recent_events(self, *, limit: int = 16) -> list[dict[str, Any]]:
        return list(self._events)[-max(1, int(limit)) :]

    def execute_manual_action(self, *, action_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "action_id": f"action::{str(action_name or '').strip()}",
            "action_name": str(action_name or "").strip(),
            "drive": 1.0,
            "reason": "manual_api",
            "params": dict(params or {}),
        }
        result = self._execute_single(payload)
        event = {"tick_index": -1, **payload, **result}
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]
        return event

    def _execute_single(self, row: dict[str, Any]) -> dict[str, Any]:
        action_name = str(row.get("action_name", "") or "")
        params = dict(row.get("params", {}) or {})
        if not self.config.enabled:
            return {"status": "disabled", "effect": "executor_disabled"}
        if self.config.dry_run or pyautogui is None:
            return {"status": "dry_run", "effect": action_name, "applied_params": params}
        try:
            if action_name == "move_mouse":
                width, height = pyautogui.size()
                x = int(_clamp(float(params.get("x", 0.5) or 0.5), 0.0, 1.0) * width)
                y = int(_clamp(float(params.get("y", 0.5) or 0.5), 0.0, 1.0) * height)
                duration = max(0.0, float(params.get("duration_sec", 0.05) or 0.05))
                pyautogui.moveTo(x, y, duration=duration)
                return {"status": "executed", "effect": "move_mouse", "applied_params": {"x": x, "y": y, "duration_sec": duration}}
            if action_name == "click":
                button = str(params.get("button", "left") or "left")
                pyautogui.click(button=button)
                return {"status": "executed", "effect": "click", "applied_params": {"button": button}}
            if action_name == "double_click":
                pyautogui.doubleClick()
                return {"status": "executed", "effect": "double_click", "applied_params": {}}
            if action_name == "scroll":
                amount = int(params.get("amount", -120) or -120)
                pyautogui.scroll(amount)
                return {"status": "executed", "effect": "scroll", "applied_params": {"amount": amount}}
            if action_name == "type_text":
                text = str(params.get("text", "") or "")
                interval_sec = max(0.0, int(self.config.type_interval_ms) / 1000.0)
                pyautogui.write(text, interval=interval_sec)
                return {"status": "executed", "effect": "type_text", "applied_params": {"chars": len(text), "interval_sec": interval_sec}}
            if action_name == "press_key":
                key = str(params.get("key", "enter") or "enter")
                pyautogui.press(key)
                return {"status": "executed", "effect": "press_key", "applied_params": {"key": key}}
            if action_name == "wait":
                duration_ms = max(0, int(params.get("duration_ms", 50) or 50))
                time.sleep(duration_ms / 1000.0)
                return {"status": "executed", "effect": "wait", "applied_params": {"duration_ms": duration_ms}}
            return {"status": "executed", "effect": action_name, "applied_params": params}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            return {"status": "error", "effect": action_name, "error": str(exc), "applied_params": params}
