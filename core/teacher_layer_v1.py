# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from typing import Any

from core.external_teacher_gateway_v1 import ExternalTeacherGatewayV1


def _round4(value: float) -> float:
    return round(float(value), 4)


class TeacherLayerV1:
    RISKY_ACTIONS = {"click", "double_click", "press_key", "type_text"}

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "heuristic",
        llm_gate_enabled: bool = True,
        llm_gate_mode: str = "heuristic",
        llm_gate_fail_open: bool = True,
        reward_scale: float = 1.0,
        punishment_scale: float = 1.0,
        repeat_window: int = 6,
        repeat_penalty: float = 0.12,
        risky_action_min_drive: float = 0.78,
        external_teacher_enabled: bool = False,
        external_teacher_mode: str = "off",
        external_teacher_stub_response_path: str = "",
        external_teacher_fail_open: bool = True,
        external_teacher_timeout_ms: int = 150,
        external_teacher_max_retries: int = 1,
        external_teacher_retry_backoff_ms: int = 25,
        external_teacher_http_endpoint: str = "",
        external_teacher_http_headers: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or "heuristic")
        self.llm_gate_enabled = bool(llm_gate_enabled)
        self.llm_gate_mode = str(llm_gate_mode or "heuristic")
        self.llm_gate_fail_open = bool(llm_gate_fail_open)
        self.reward_scale = max(0.0, float(reward_scale))
        self.punishment_scale = max(0.0, float(punishment_scale))
        self.repeat_window = max(1, int(repeat_window))
        self.repeat_penalty = max(0.0, float(repeat_penalty))
        self.risky_action_min_drive = max(0.0, float(risky_action_min_drive))
        self._recent_action_names: deque[str] = deque(maxlen=self.repeat_window)
        self.external_teacher_enabled = bool(external_teacher_enabled)
        self.external_teacher_fail_open = bool(external_teacher_fail_open)
        self.external_teacher_gateway = ExternalTeacherGatewayV1(
            mode=str(external_teacher_mode or "off"),
            stub_response_path=str(external_teacher_stub_response_path or ""),
            max_retries=int(external_teacher_max_retries),
            retry_backoff_ms=int(external_teacher_retry_backoff_ms),
            timeout_ms=int(external_teacher_timeout_ms),
            http_endpoint=str(external_teacher_http_endpoint or ""),
            http_headers=dict(external_teacher_http_headers or {}),
        )

    def review_actions(
        self,
        *,
        tick_index: int,
        action_drives: list[dict[str, Any]],
        runtime_tick: dict[str, Any],
        autonomous_state: dict[str, int] | None = None,
        teacher_mode_override: str | None = None,
        llm_gate_mode_override: str | None = None,
        external_teacher_enabled_override: bool | None = None,
        external_teacher_mode_override: str | None = None,
        external_teacher_stub_response_path_override: str | None = None,
        external_teacher_fail_open_override: bool | None = None,
        external_teacher_max_retries_override: int | None = None,
        external_teacher_retry_backoff_ms_override: int | None = None,
        external_teacher_http_endpoint_override: str | None = None,
        external_teacher_http_headers_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action_drives = [dict(item) for item in (action_drives or []) if isinstance(item, dict)]
        candidate_pool = [dict(item) for item in action_drives]
        planner_selected = [dict(item) for item in action_drives if bool(item.get("planner_selected", False))]
        if planner_selected:
            action_drives = planner_selected
        mode = str(teacher_mode_override or self.mode or "heuristic")
        llm_gate_mode = str(llm_gate_mode_override or self.llm_gate_mode or "heuristic")
        if not self.enabled or mode == "off":
            return {
                "applied": False,
                "mode": mode,
                "scored_action_drives": action_drives,
                "candidate_action_drives": candidate_pool,
                "planner_selected_action_drives": planner_selected,
                "blocked_actions": [],
                "warnings": [],
                "notes": ["teacher_disabled"],
            }

        state_summary = dict(runtime_tick.get("state_pool_summary", {}) or {})
        residual_count = int(((state_summary.get("residual_summary") or {}).get("count", 0)) or 0)
        selected: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        external_review = self._review_with_external_teacher(
            tick_index=tick_index,
            action_drives=action_drives,
            runtime_tick=runtime_tick,
            external_teacher_enabled_override=external_teacher_enabled_override,
            external_teacher_mode_override=external_teacher_mode_override,
            external_teacher_stub_response_path_override=external_teacher_stub_response_path_override,
            external_teacher_fail_open_override=external_teacher_fail_open_override,
            external_teacher_max_retries_override=external_teacher_max_retries_override,
            external_teacher_retry_backoff_ms_override=external_teacher_retry_backoff_ms_override,
            external_teacher_http_endpoint_override=external_teacher_http_endpoint_override,
            external_teacher_http_headers_override=external_teacher_http_headers_override,
        )

        for raw in action_drives:
            row = dict(raw)
            action_id = str(row.get("action_id", "") or "")
            action_name = str(row.get("action_name", "") or action_id.replace("action::", "")).strip()
            drive = float(row.get("drive", 0.0) or 0.0)
            firmness = float(row.get("firmness", 0.0) or 0.0)
            effective_drive = drive + max(0.0, firmness) * 2.0
            row["action_name"] = action_name
            teacher_penalty = 0.0
            teacher_reward = 0.0
            notes: list[str] = []
            blocked_reason = ""

            recent_repeat_count = sum(1 for item in self._recent_action_names if item == action_name)
            if recent_repeat_count > 0:
                teacher_penalty += min(self.repeat_penalty, recent_repeat_count * self.repeat_penalty / max(1, self.repeat_window))
                notes.append("repeat_penalty")

            if action_name in self.RISKY_ACTIONS and effective_drive < self.risky_action_min_drive:
                blocked_reason = "risky_action_drive_too_low"
            elif action_name in {"click", "double_click"} and residual_count > 0:
                blocked_reason = "residual_unresolved"
            elif action_name in {"type_text", "press_key"} and int((autonomous_state or {}).get("idle_ticks", 0) or 0) > max(2, self.repeat_window // 2):
                teacher_penalty += 0.08
                notes.append("idle_context_penalty")

            if action_name in {"continue_focus", "inspect_residual", "move_gaze"}:
                teacher_reward += 0.04
                notes.append("ap_focus_preserved")

            final_drive = max(0.0, min(1.5, drive + teacher_reward - teacher_penalty))
            row["teacher_reward"] = _round4(teacher_reward)
            row["teacher_penalty"] = _round4(teacher_penalty)
            row["drive_before_teacher"] = _round4(drive)
            row["teacher_effective_drive"] = _round4(effective_drive)
            row["drive"] = _round4(final_drive)
            row["teacher_notes"] = notes

            gate = self._llm_gate_stub_with_mode(
                action_name=action_name,
                drive=max(final_drive, effective_drive),
                residual_count=residual_count,
                llm_gate_mode_override=llm_gate_mode,
            )
            row["llm_gate"] = gate
            if not bool(gate.get("allow", True)):
                blocked_reason = str(gate.get("reason", "") or blocked_reason or "llm_gate_blocked")
            if bool(external_review.get("fail_closed", False)) and action_name in self.RISKY_ACTIONS and not blocked_reason:
                blocked_reason = "external_teacher_unavailable_fail_closed"

            external_decision = self._match_external_decision(external_review=external_review, row=row)
            if external_decision:
                row["external_teacher"] = external_decision
                teacher_reward += float(external_decision.get("reward", 0.0) or 0.0)
                teacher_penalty += float(external_decision.get("punishment", 0.0) or 0.0)
                final_drive = max(0.0, min(1.5, drive + teacher_reward - teacher_penalty))
                row["teacher_reward"] = _round4(teacher_reward)
                row["teacher_penalty"] = _round4(teacher_penalty)
                row["drive"] = _round4(final_drive)
                decision = str(external_decision.get("decision", "allow") or "allow")
                if decision == "block":
                    blocked_reason = str(external_decision.get("warning_code", "") or "external_teacher_blocked")
                elif decision == "warn":
                    warnings.append(
                        {
                            "code": str(external_decision.get("warning_code", "") or "external_teacher_warn"),
                            "action_id": action_id,
                            "action_name": action_name,
                            "source": "external_teacher",
                        }
                    )

            if blocked_reason:
                blocked.append({**row, "blocked_reason": blocked_reason})
                warnings.append({"code": blocked_reason, "action_id": action_id, "action_name": action_name})
                continue
            selected.append(row)

        selected.sort(key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or "")))
        top_action = selected[0]["action_name"] if selected else ""
        if top_action:
            self._recent_action_names.append(top_action)
        return {
            "applied": True,
            "mode": mode,
            "llm_gate_mode": llm_gate_mode,
            "scored_action_drives": selected,
            "candidate_action_drives": candidate_pool,
            "planner_selected_action_drives": planner_selected,
            "blocked_actions": blocked,
            "warnings": warnings,
            "external_teacher_review": external_review,
            "notes": ["teacher_review_applied"],
        }

    def build_teacher_feedback(
        self,
        *,
        tick_index: int,
        runtime_tick: dict[str, Any],
        teacher_review: dict[str, Any],
        selected_actions: list[dict[str, Any]],
        sandbox_result: dict[str, Any],
        runtime_action_effects: dict[str, Any],
    ) -> dict[str, Any]:
        mode = str((teacher_review or {}).get("mode", self.mode) or self.mode)
        llm_gate_mode = str((teacher_review or {}).get("llm_gate_mode", self.llm_gate_mode) or self.llm_gate_mode)
        if not self.enabled or mode == "off":
            return {}

        reward = 0.0
        punishment = 0.0
        notes: list[str] = []
        emotion_channels = dict((runtime_tick.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
        expectation = float(emotion_channels.get("expectation", 0.0) or 0.0)
        pressure = float(emotion_channels.get("pressure", 0.0) or 0.0)
        dissonance = float(emotion_channels.get("dissonance", 0.0) or 0.0)
        correctness = float(emotion_channels.get("correctness", 0.0) or 0.0)
        blocked_actions = list(teacher_review.get("blocked_actions", []) or [])
        external_review = dict(teacher_review.get("external_teacher_review", {}) or {})
        external_decisions = [dict(item) for item in (external_review.get("decisions", []) or []) if isinstance(item, dict)]

        if selected_actions:
            reward += 0.03
            notes.append("teacher_observed_action")
        else:
            punishment += 0.03
            notes.append("teacher_no_action")

        if bool(runtime_action_effects.get("moved", False)):
            reward += 0.04
            notes.append("teacher_action_effective")

        if blocked_actions:
            reward += 0.02
            notes.append("teacher_blocked_risky_action")

        for decision in external_decisions:
            reward += float(decision.get("reward", 0.0) or 0.0)
            punishment += float(decision.get("punishment", 0.0) or 0.0)
            decision_kind = str(decision.get("decision", "") or "")
            if decision_kind == "block":
                notes.append("external_teacher_block")
            elif decision_kind == "warn":
                notes.append("external_teacher_warn")

        if expectation > pressure and correctness >= dissonance:
            reward += 0.03
            notes.append("teacher_alignment_positive")
        if dissonance > correctness + 0.2:
            punishment += 0.04
            notes.append("teacher_alignment_negative")

        payload: dict[str, Any] = {}
        reward *= self.reward_scale
        punishment *= self.punishment_scale
        if reward > 0:
            payload["reward"] = _round4(reward)
        if punishment > 0:
            payload["punishment"] = _round4(punishment)
        if notes:
            payload["notes"] = notes
        payload["teacher_review"] = {
            "blocked_count": len(blocked_actions),
            "selected_count": len(selected_actions),
            "mode": mode,
            "llm_gate_mode": llm_gate_mode,
            "external_teacher_mode": str(external_review.get("mode", "") or ""),
            "external_teacher_decision_count": len(external_decisions),
        }
        if external_decisions:
            payload["external_teacher_review"] = external_review
        return payload

    def export_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "llm_gate_enabled": self.llm_gate_enabled,
            "llm_gate_mode": self.llm_gate_mode,
            "llm_gate_fail_open": self.llm_gate_fail_open,
            "reward_scale": self.reward_scale,
            "punishment_scale": self.punishment_scale,
            "repeat_window": self.repeat_window,
            "repeat_penalty": self.repeat_penalty,
            "risky_action_min_drive": self.risky_action_min_drive,
            "external_teacher_enabled": self.external_teacher_enabled,
            "external_teacher_fail_open": self.external_teacher_fail_open,
            "external_teacher_gateway": self.external_teacher_gateway.export_payload(),
            "recent_action_names": list(self._recent_action_names),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._recent_action_names = deque(
            [str(item or "") for item in (payload.get("recent_action_names", []) or []) if str(item or "")],
            maxlen=self.repeat_window,
        )
        self.external_teacher_enabled = bool(payload.get("external_teacher_enabled", self.external_teacher_enabled))
        self.external_teacher_fail_open = bool(payload.get("external_teacher_fail_open", self.external_teacher_fail_open))
        gateway_payload = payload.get("external_teacher_gateway")
        if isinstance(gateway_payload, dict):
            self.external_teacher_gateway.import_payload(gateway_payload)

    def _llm_gate_stub_with_mode(
        self,
        *,
        action_name: str,
        drive: float,
        residual_count: int,
        llm_gate_mode_override: str | None,
    ) -> dict[str, Any]:
        llm_gate_mode = str(llm_gate_mode_override or self.llm_gate_mode or "heuristic")
        if not self.llm_gate_enabled or llm_gate_mode == "off":
            return {"allow": True, "source": "off", "reason": ""}
        try:
            if llm_gate_mode == "heuristic":
                if action_name in self.RISKY_ACTIONS and drive < self.risky_action_min_drive:
                    return {"allow": False, "source": "heuristic", "reason": "llm_stub_low_confidence_risky_action"}
                if action_name in {"click", "double_click"} and residual_count > 0:
                    return {"allow": False, "source": "heuristic", "reason": "llm_stub_wait_for_clearer_state"}
                return {"allow": True, "source": "heuristic", "reason": ""}
            if llm_gate_mode == "stub_file":
                return {"allow": True, "source": "stub_file", "reason": ""}
        except Exception as exc:
            if self.llm_gate_fail_open:
                return {"allow": True, "source": "fail_open", "reason": str(exc)}
            return {"allow": False, "source": "fail_closed", "reason": str(exc)}
        return {"allow": True, "source": llm_gate_mode, "reason": ""}

    def _review_with_external_teacher(
        self,
        *,
        tick_index: int,
        action_drives: list[dict[str, Any]],
        runtime_tick: dict[str, Any],
        external_teacher_enabled_override: bool | None = None,
        external_teacher_mode_override: str | None = None,
        external_teacher_stub_response_path_override: str | None = None,
        external_teacher_fail_open_override: bool | None = None,
        external_teacher_max_retries_override: int | None = None,
        external_teacher_retry_backoff_ms_override: int | None = None,
        external_teacher_http_endpoint_override: str | None = None,
        external_teacher_http_headers_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        enabled = self.external_teacher_enabled if external_teacher_enabled_override is None else bool(external_teacher_enabled_override)
        if not enabled:
            return {"applied": False, "mode": "off", "decisions": [], "warnings": []}
        fail_open = self.external_teacher_fail_open if external_teacher_fail_open_override is None else bool(external_teacher_fail_open_override)
        mode = str(external_teacher_mode_override or self.external_teacher_gateway.mode or "off")
        stub_path = str(external_teacher_stub_response_path_override or self.external_teacher_gateway.stub_response_path or "")
        http_endpoint = str(external_teacher_http_endpoint_override or self.external_teacher_gateway.http_endpoint or "")
        max_retries = (
            self.external_teacher_gateway.max_retries
            if external_teacher_max_retries_override is None
            else max(1, int(external_teacher_max_retries_override))
        )
        retry_backoff_ms = (
            self.external_teacher_gateway.retry_backoff_ms
            if external_teacher_retry_backoff_ms_override is None
            else max(0, int(external_teacher_retry_backoff_ms_override))
        )
        http_headers = (
            dict(external_teacher_http_headers_override or {})
            if external_teacher_http_headers_override is not None
            else dict(self.external_teacher_gateway.http_headers or {})
        )
        gateway = ExternalTeacherGatewayV1(
            mode=mode,
            stub_response_path=stub_path,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
            timeout_ms=self.external_teacher_gateway.timeout_ms,
            http_endpoint=http_endpoint,
            http_headers=http_headers,
        )
        request_payload = {
            "tick_index": int(tick_index),
            "candidate_actions": [
                {
                    "action_id": str(item.get("action_id", "") or ""),
                    "action_name": str(item.get("action_name", "") or "") or str(item.get("action_id", "") or "").replace("action::", ""),
                    "drive": _round4(float(item.get("drive", 0.0) or 0.0)),
                }
                for item in (action_drives or [])
                if isinstance(item, dict)
            ],
            "focus_preview": list(((runtime_tick.get("a_focus", {}) or {}).get("focus_units", []) or [])[:6]),
            "bn_preview_ids": [str(item.get("memory_id", "") or "") for item in (runtime_tick.get("bn_list", []) or [])[:6]],
            "state_summary": dict(runtime_tick.get("state_pool_summary", {}) or {}),
        }
        response = gateway.review(request_payload=request_payload)
        if not bool(response.get("ok", False)):
            warnings = []
            error_code = str(response.get("error", "") or "external_teacher_error")
            if error_code:
                warnings.append({"code": error_code, "source": "external_teacher"})
            return {
                "applied": False,
                "mode": str(response.get("mode", "") or gateway.mode),
                "error": str(response.get("error", "") or ""),
                "fail_closed": not fail_open,
                "fail_open": fail_open,
                "provider": str(response.get("provider", "") or gateway.mode),
                "reviewer": str(response.get("reviewer", "") or gateway.mode),
                "transport_audit": dict(response.get("transport_audit", {}) or {}),
                "decisions": [],
                "warnings": warnings,
            }
        warnings = []
        for item in response.get("decisions", []) or []:
            if not isinstance(item, dict):
                continue
            decision_kind = str(item.get("decision", "") or "")
            if decision_kind in {"warn", "block"}:
                warnings.append(
                    {
                        "code": str(item.get("warning_code", "") or f"external_teacher_{decision_kind}"),
                        "action_id": str(item.get("target_action_id", "") or ""),
                        "action_name": str(item.get("target_action_name", "") or ""),
                        "source": "external_teacher",
                    }
                )
        return {
            "applied": True,
            "mode": str(response.get("mode", "") or gateway.mode),
            "reviewer": str(response.get("reviewer", "") or ""),
            "path": str(response.get("path", "") or ""),
            "request_digest": dict(response.get("request_digest", {}) or {}),
            "provider": str(response.get("provider", "") or gateway.mode),
            "fail_closed": False,
            "fail_open": fail_open,
            "transport_audit": dict(response.get("transport_audit", {}) or {}),
            "decisions": [dict(item) for item in (response.get("decisions", []) or []) if isinstance(item, dict)],
            "warnings": warnings,
        }

    def _match_external_decision(self, *, external_review: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        row_action_id = str(row.get("action_id", "") or "")
        row_action_name = str(row.get("action_name", "") or "")
        for item in external_review.get("decisions", []) or []:
            if not isinstance(item, dict):
                continue
            target_action_id = str(item.get("target_action_id", "") or "")
            target_action_name = str(item.get("target_action_name", "") or "")
            if target_action_id and target_action_id == row_action_id:
                return dict(item)
            if target_action_name and target_action_name == row_action_name:
                return dict(item)
        return {}
