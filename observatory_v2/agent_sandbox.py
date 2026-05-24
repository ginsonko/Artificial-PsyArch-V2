# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from .computer_executor import ComputerExecutorV2, ExecutorConfig


class AgentSandboxV1:
    def __init__(
        self,
        *,
        enabled: bool = False,
        dry_run: bool = True,
        max_actions_per_tick: int = 1,
        screenshot_enabled: bool = False,
        screenshot_scale: float = 0.75,
        type_interval_ms: int = 15,
        max_events: int = 256,
    ) -> None:
        self._executor = ComputerExecutorV2(
            config=ExecutorConfig(
                enabled=bool(enabled),
                dry_run=bool(dry_run),
                max_actions_per_tick=max(1, int(max_actions_per_tick)),
                screenshot_enabled=bool(screenshot_enabled),
                screenshot_scale=float(screenshot_scale),
                type_interval_ms=max(0, int(type_interval_ms)),
            ),
            max_events=max_events,
        )

    def evaluate_action_drives(self, *, tick_index: int, action_drives: list[dict[str, Any]]) -> dict[str, Any]:
        return self._executor.evaluate_and_execute(tick_index=tick_index, action_drives=action_drives)

    def capture_screenshot_packet(self, *, force: bool = False) -> dict[str, Any]:
        return self._executor.capture_screenshot_packet(force=force)

    def export_payload(self) -> dict[str, Any]:
        return self._executor.export_payload()

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._executor.import_payload(payload)

    def status(self) -> dict[str, Any]:
        return self._executor.status()

    def recent_events(self, *, limit: int = 16) -> list[dict[str, Any]]:
        return self._executor.recent_events(limit=limit)

    def execute_manual_action(self, *, action_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._executor.execute_manual_action(action_name=action_name, params=params)
