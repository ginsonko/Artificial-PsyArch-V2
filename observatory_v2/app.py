# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from core.runtime_v2 import RuntimeV2
from sensors.hearing_sensor_v1 import HearingSensorV1
from sensors.stream_adapter_v1 import BaseRealtimeSourceV1, StreamAdapterV1
from sensors.vision_sensor_v1 import VisionSensorV1
from .agent_sandbox import AgentSandboxV1
from .config import AppConfig, load_config, repo_root
from .io_utils import append_jsonl, iter_jsonl, read_json, write_json
from .run_rollup import empty_rollup, update_rollup
from .schema_tools import load_schema, validate_or_raise
from .storage import (
    StorageLayout,
    build_storage_layout,
    chunk_file,
    list_runs,
    make_run_dir,
    make_run_id,
    now_ms,
    overlay_manifest_with_session_status,
    read_autonomous_session_status,
)


class AppError(RuntimeError):
    pass


class GracefulRunStop(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason or "run stopped")
        self.details = dict(details or {})


class ObservatoryV2App:
    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        repo_root_value: Path | None = None,
        outputs_root_override: str | None = None,
    ) -> None:
        self.repo_root = (repo_root_value or repo_root()).resolve()
        self.config = config or load_config()
        self._service_started_at_ms = now_ms()
        self._service_boot_id = f"observatory_v2::{self._service_started_at_ms}::{os.getpid()}"
        outputs_root_value = outputs_root_override if outputs_root_override is not None else self.config.outputs_root
        self.layout: StorageLayout = build_storage_layout(self.repo_root, outputs_root_value)
        self._lock = threading.RLock()
        self._active_thread: threading.Thread | None = None
        self._active_run_id = ""
        self._active_session_id = ""
        self._autonomous_session_status: dict[str, Any] = {}
        self._autonomous_session_pause_event: threading.Event | None = None
        self._autonomous_session_stop_event: threading.Event | None = None
        self._active_stream_source: BaseRealtimeSourceV1 | None = None
        self._last_forget_summary: dict[str, Any] = {}
        self._last_forget_preview_summary: dict[str, Any] = {}
        self._live_ring: deque[dict[str, Any]] = deque(maxlen=self.config.live_ring_limit)
        self._runtime = self._build_runtime()
        self._agent_sandbox = self._build_agent_sandbox()
        self._stream_adapter = StreamAdapterV1()
        self._latest_live: dict[str, Any] = {
            "schema_id": "live_snapshot/v1",
            "status": "idle",
            "active_run_id": "",
            "recent_ticks": [],
            "server_time_ms": now_ms(),
        }
        self._bootstrap_service_runtime_state_from_disk()
        self._repair_bootstrap_autonomous_session_statuses()
        self._bootstrap_latest_live_from_disk()

    def _run_asset_store_dir(self, run_dir: Path) -> Path:
        path = run_dir / "assets_store"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _store_text_asset(self, run_dir: Path, *, category: str, text: str) -> dict[str, Any]:
        clean_text = str(text or "")
        digest = hashlib.blake2b(clean_text.encode("utf-8"), digest_size=16).hexdigest()
        rel_path = Path(str(category or "misc")) / f"{digest}.txt"
        asset_path = self._run_asset_store_dir(run_dir) / rel_path
        if not asset_path.exists():
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(clean_text, encoding="utf-8")
        return {
            "schema_id": "asset_ref/v1",
            "asset_kind": "utf8_text",
            "rel_path": rel_path.as_posix(),
            "length": len(clean_text),
        }

    def _read_text_asset(self, run_dir: Path, ref: dict[str, Any]) -> str:
        rel_path = str(ref.get("rel_path", "") or "").strip()
        if not rel_path:
            return ""
        asset_path = self._run_asset_store_dir(run_dir) / Path(rel_path)
        if not asset_path.exists():
            return ""
        try:
            return asset_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def get_run_text_asset(self, run_id: str, rel_path: str) -> str:
        run_dir = self.layout.runs_root / str(run_id or "")
        clean_rel_path = str(rel_path or "").strip()
        if not clean_rel_path:
            return ""
        ref = {
            "schema_id": "asset_ref/v1",
            "asset_kind": "utf8_text",
            "rel_path": clean_rel_path,
        }
        return self._read_text_asset(run_dir, ref)

    def _compact_media_assets_for_storage(self, run_dir: Path, payload: Any, *, category: str) -> Any:
        if isinstance(payload, dict):
            if str(payload.get("schema_id", "") or "") == "asset_ref/v1":
                return dict(payload)
            compacted: dict[str, Any] = {}
            for key, value in payload.items():
                next_category = f"{category}/{key}" if category else str(key)
                if isinstance(value, str):
                    is_media_blob = (
                        value.startswith("data:image/")
                        or (key in {"preview_wav_b64", "proxy_preview_wav_b64"} and len(value) >= 2048)
                    )
                    if is_media_blob:
                        compacted[key] = self._store_text_asset(run_dir, category=next_category, text=value)
                    else:
                        compacted[key] = value
                else:
                    compacted[key] = self._compact_media_assets_for_storage(run_dir, value, category=next_category)
            return compacted
        if isinstance(payload, list):
            return [self._compact_media_assets_for_storage(run_dir, item, category=category) for item in payload]
        return payload

    def _restore_media_assets_from_storage(self, run_dir: Path, payload: Any) -> Any:
        if isinstance(payload, dict):
            if str(payload.get("schema_id", "") or "") == "asset_ref/v1":
                return self._read_text_asset(run_dir, payload)
            return {key: self._restore_media_assets_from_storage(run_dir, value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._restore_media_assets_from_storage(run_dir, item) for item in payload]
        return payload

    def _storage_preview_neighbors(self, neighbors: list[dict[str, Any]] | None, *, limit: int = 3) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in list(neighbors or [])[: max(0, int(limit))]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "memory_id": str(item.get("memory_id", "") or ""),
                    "text": str(item.get("text", "") or ""),
                    "score": round(float(item.get("score", 0.0) or 0.0), 4),
                    "successor_bias": round(float(item.get("successor_bias", 0.0) or 0.0), 4),
                }
            )
        return rows

    def _storage_preview_c_items(self, items: list[dict[str, Any]] | None, *, limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in list(items or [])[: max(0, int(limit))]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "sa_label": str(item.get("sa_label", "") or ""),
                    "display_text": str(item.get("display_text", "") or ""),
                    "energy": round(float(item.get("energy", 0.0) or 0.0), 4),
                    "source_type": str(item.get("source_type", "") or ""),
                }
            )
        return rows

    def _compact_bn_list_for_storage(self, bn_list: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in list(bn_list or []):
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "memory_id": str(item.get("memory_id", "") or ""),
                    "memory_kind": str(item.get("memory_kind", "") or ""),
                    "tick_index": int(item.get("tick_index", -1) or -1),
                    "text": str(item.get("text", "") or ""),
                    "memory_modalities": [str(x or "") for x in (item.get("memory_modalities", []) or []) if str(x or "")],
                    "raw_score": round(float(item.get("raw_score", 0.0) or 0.0), 4),
                    "score": round(float(item.get("score", 0.0) or 0.0), 4),
                    "overlap_labels": [str(x or "") for x in (item.get("overlap_labels", []) or [])[:12] if str(x or "")],
                    "overlap_label_count": len(item.get("overlap_labels", []) or []),
                    "candidate_sources": [str(x or "") for x in (item.get("candidate_sources", []) or []) if str(x or "")],
                    "vector_tokens": [str(x or "") for x in (item.get("vector_tokens", []) or [])[:12] if str(x or "")],
                    "query_vector_tokens": [str(x or "") for x in (item.get("query_vector_tokens", []) or [])[:12] if str(x or "")],
                    "score_breakdown": dict(item.get("score_breakdown", {}) or {}),
                    "vector_engine": str(item.get("vector_engine", "") or ""),
                }
            )
        return compact

    def _compact_c_i_list_for_storage(self, c_i_list: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in list(c_i_list or []):
            if not isinstance(item, dict):
                continue
            neighbors = list(item.get("neighbors", []) or [])
            branch_items = list(item.get("items", []) or [])
            compact.append(
                {
                    "bundle_id": str(item.get("bundle_id", "") or ""),
                    "source_bn_id": str(item.get("source_bn_id", "") or ""),
                    "virtual_energy": round(float(item.get("virtual_energy", 0.0) or 0.0), 4),
                    "credibility_key": str(item.get("credibility_key", "") or ""),
                    "credibility_multiplier": round(float(item.get("credibility_multiplier", 0.0) or 0.0), 4),
                    "credibility_bias": round(float(item.get("credibility_bias", 0.0) or 0.0), 4),
                    "neighbor_count": len(neighbors),
                    "neighbors": self._storage_preview_neighbors(neighbors),
                    "items_count": len(branch_items),
                    "items": self._storage_preview_c_items(branch_items),
                }
            )
        return compact

    def _build_tick_cache_row(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "tick_index": int(summary.get("tick_index", -1) or -1),
            "tick_id": str(summary.get("tick_id", "") or ""),
            "generated_at_ms": int(summary.get("generated_at_ms", 0) or 0),
            "input_preview": str(summary.get("input_preview", "") or ""),
            "a_focus_preview": [str(item or "") for item in (summary.get("a_focus_preview", []) or [])[:8] if str(item or "")],
            "status": str(summary.get("status", "") or "ok"),
            "bn_preview": list(summary.get("bn_preview", []) or []),
            "multimodal_summary": dict(summary.get("multimodal_summary", {}) or {}),
            "rules_preview": dict(summary.get("rules_preview", {}) or {}),
        }

    def _build_runtime(self) -> RuntimeV2:
        return RuntimeV2(config=self.config, repo_root=self.repo_root)

    def _build_agent_sandbox(self) -> AgentSandboxV1:
        return AgentSandboxV1(
            enabled=self.config.executor_enabled,
            dry_run=self.config.executor_dry_run,
            max_actions_per_tick=self.config.executor_max_actions_per_tick,
            screenshot_enabled=self.config.executor_screenshot_enabled,
            screenshot_scale=self.config.executor_screenshot_scale,
            type_interval_ms=self.config.executor_type_interval_ms,
        )

    def _service_runtime_state_path(self) -> Path:
        return self.layout.outputs_root / "live" / "service_runtime_state.json"

    def _bootstrap_service_runtime_state_from_disk(self) -> None:
        payload = read_json(self._service_runtime_state_path(), default={})
        if not isinstance(payload, dict):
            return
        last_forget_summary = payload.get("last_forget_summary")
        if isinstance(last_forget_summary, dict):
            self._last_forget_summary = copy.deepcopy(last_forget_summary)
        last_forget_preview_summary = payload.get("last_forget_preview_summary")
        if isinstance(last_forget_preview_summary, dict):
            self._last_forget_preview_summary = copy.deepcopy(last_forget_preview_summary)

    def _persist_service_runtime_state(self) -> None:
        payload = {
            "schema_id": "observatory_service_runtime_state/v1",
            "schema_version": "1.0",
            "updated_at_ms": now_ms(),
            "latest_run_id": self.latest_run_id(),
            "last_forget_summary": copy.deepcopy(self._last_forget_summary),
            "last_forget_preview_summary": copy.deepcopy(self._last_forget_preview_summary),
        }
        write_json(self._service_runtime_state_path(), payload)

    def _build_autonomous_session_goal(
        self,
        *,
        label: str,
        text_hint: str,
        max_ticks: int | None,
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        ts_ms = int(now_ts or now_ms())
        clean_label = str(label or "Autonomous Session").strip() or "Autonomous Session"
        clean_text_hint = str(text_hint or "").strip()
        return {
            "label": clean_label,
            "goal_text": clean_text_hint or clean_label,
            "goal_source": "text_hint" if clean_text_hint else "label",
            "target_kind": "bounded_ticks" if int(max_ticks or 0) > 0 else "open_ended",
            "target_tick_count": int(max_ticks or 0),
            "phase_label": "queued_wait",
            "phase_index": 0,
            "phase_status": "queued",
            "ticks_completed": 0,
            "remaining_tick_budget": int(max_ticks or 0),
            "completion_ratio": 0.0,
            "focus_preview": [],
            "selected_action_names": [],
            "recover_hint": "",
            "updated_at_ms": ts_ms,
        }

    def _build_autonomous_session_lifecycle(self, *, created_at_ms: int, initial_status: str = "queued") -> dict[str, Any]:
        return {
            "start_count": 1,
            "pause_request_count": 0,
            "paused_count": 0,
            "resume_count": 0,
            "recover_count": 0,
            "stop_request_count": 0,
            "interrupt_count": 0,
            "completion_count": 0,
            "failure_count": 0,
            "transition_count": 1,
            "last_transition": str(initial_status or "queued"),
            "last_transition_at_ms": int(created_at_ms),
            "last_transition_reason": "",
            "last_status": str(initial_status or "queued"),
        }

    def _build_autonomous_session_context(self, *, now_ts: int | None = None) -> dict[str, Any]:
        ts_ms = int(now_ts or now_ms())
        return {
            "last_tick_id": "",
            "last_input_preview": "",
            "last_focus_preview": [],
            "last_bn_ids": [],
            "last_selected_action_names": [],
            "last_selected_action_statuses": [],
            "last_teacher_mode": "",
            "last_external_teacher_mode": "",
            "updated_at_ms": ts_ms,
        }

    def _build_autonomous_session_health(self, *, now_ts: int | None = None) -> dict[str, Any]:
        ts_ms = int(now_ts or now_ms())
        return {
            "health_status": "queued",
            "health_reason": "queued_waiting_for_first_tick",
            "recover_hint": "",
            "idle_ticks": 0,
            "capture_failures": 0,
            "action_errors": 0,
            "last_logic_ms": 0.0,
            "last_sleep_ms": 0,
            "last_screen_capture_ok": False,
            "last_focus_preview": [],
            "last_input_preview": "",
            "last_selected_action_names": [],
            "last_selected_action_statuses": [],
            "last_bn_ids": [],
            "last_tick_generated_at_ms": 0,
            "last_checkpoint_at_ms": 0,
            "last_checkpoint_tick_done": 0,
            "updated_at_ms": ts_ms,
        }

    def _infer_autonomous_session_phase_label(self, status: dict[str, Any]) -> str:
        current_status = str(status.get("status", "") or "").strip() or "queued"
        health = dict(status.get("session_health", {}) or {})
        context = dict(status.get("session_context", {}) or {})
        if current_status == "queued":
            return "queued_wait"
        if current_status == "recovering":
            return "recover_resume"
        if current_status == "pausing":
            return "pause_pending"
        if current_status == "paused":
            return "paused_wait"
        if current_status == "stopping":
            return "shutdown_pending"
        if current_status in {"completed", "stopped", "failed", "interrupted"}:
            return current_status
        if int(health.get("capture_failures", 0) or 0) > 0:
            return "capture_retry"
        if int(health.get("action_errors", 0) or 0) > 0:
            return "action_recovery"
        if int(health.get("idle_ticks", 0) or 0) > 0:
            return "idle_backoff"
        if list(context.get("last_selected_action_names", []) or []):
            return "action_followthrough"
        if list(context.get("last_focus_preview", []) or []):
            return "focus_growth"
        return "autonomous_exploration"

    def _update_autonomous_session_health_fields(self, status: dict[str, Any]) -> None:
        if not isinstance(status, dict):
            return
        health = dict(status.get("session_health", {}) or {})
        context = dict(status.get("session_context", {}) or {})
        current_status = str(status.get("status", "") or "").strip() or "queued"
        active = bool(status.get("active", False))
        recoverable = bool(status.get("recoverable", False))
        max_ticks = int(status.get("max_ticks", 0) or 0)
        tick_done = int(status.get("tick_done", 0) or 0)
        remaining = max(0, max_ticks - tick_done) if max_ticks > 0 else 0
        idle_ticks = int(health.get("idle_ticks", 0) or 0)
        capture_failures = int(health.get("capture_failures", 0) or 0)
        action_errors = int(health.get("action_errors", 0) or 0)
        last_selected_action_names = [str(item or "") for item in (context.get("last_selected_action_names", []) or []) if str(item or "")]
        last_focus_preview = [str(item or "") for item in (context.get("last_focus_preview", []) or []) if str(item or "")]

        health_status = "idle"
        health_reason = "session_idle"
        if current_status == "queued":
            health_status = "queued"
            health_reason = "queued_waiting_for_first_tick"
        elif current_status == "recovering":
            health_status = "recovering"
            health_reason = "recovering_from_runtime_checkpoint"
        elif current_status == "pausing":
            health_status = "pause_pending"
            health_reason = "pause_requested_waiting_barrier"
        elif current_status == "paused":
            health_status = "paused"
            health_reason = "session_paused_by_operator"
        elif current_status == "stopping":
            health_status = "stopping"
            health_reason = "stop_requested_waiting_cleanup"
        elif current_status == "completed":
            health_status = "completed"
            health_reason = "target_completed"
        elif current_status == "stopped":
            health_status = "stopped"
            health_reason = str(status.get("last_stop_reason", "") or "session_stopped")
        elif current_status == "failed":
            health_status = "failed"
            health_reason = str(status.get("last_stop_reason", "") or "session_failed")
        elif current_status == "interrupted":
            health_status = "interrupted"
            health_reason = str(status.get("last_stop_reason", "") or "session_interrupted")
        elif capture_failures > 0:
            health_status = "capture_warning"
            health_reason = "recent_capture_failures"
        elif action_errors > 0:
            health_status = "action_warning"
            health_reason = "recent_action_errors"
        elif idle_ticks > 0:
            health_status = "idle_warning"
            health_reason = "recent_idle_ticks"
        elif active and last_selected_action_names:
            health_status = "engaged"
            health_reason = "actions_selected_recently"
        elif active and last_focus_preview:
            health_status = "healthy"
            health_reason = "focus_chain_advancing"
        elif active:
            health_status = "healthy"
            health_reason = "session_running"

        if recoverable and current_status in {"stopped", "failed", "interrupted"}:
            if max_ticks > 0:
                recover_hint = f"可从本地 tick {tick_done} 继续，剩余预算 {remaining}"
            else:
                recover_hint = f"可从本地 tick {tick_done} 继续，当前是开放式 session"
        elif current_status in {"paused", "pausing"}:
            if max_ticks > 0:
                recover_hint = f"可直接恢复继续推进，当前进度 {tick_done}/{max_ticks}"
            else:
                recover_hint = f"可直接恢复继续推进，当前已完成 {tick_done} tick"
        elif current_status == "completed":
            recover_hint = "已完成，无需恢复"
        elif current_status == "recovering":
            recover_hint = f"正在从 checkpoint 恢复，准备继续本地 tick {tick_done}"
        elif current_status in {"running", "queued"}:
            recover_hint = "当前 session 正在推进，可继续观察或稍后暂停"
        else:
            recover_hint = ""

        health["health_status"] = health_status
        health["health_reason"] = health_reason
        health["recover_hint"] = recover_hint
        health["last_focus_preview"] = last_focus_preview[:8]
        health["last_input_preview"] = str(context.get("last_input_preview", "") or "")[:160]
        health["last_selected_action_names"] = last_selected_action_names[:8]
        health["last_selected_action_statuses"] = [
            str(item or "")
            for item in (context.get("last_selected_action_statuses", []) or [])
            if str(item or "")
        ][:8]
        health["last_bn_ids"] = [
            str(item or "")
            for item in (context.get("last_bn_ids", []) or [])
            if str(item or "")
        ][:8]
        health["updated_at_ms"] = int(status.get("updated_at_ms", now_ms()) or now_ms())
        status["session_health"] = health

    def _update_autonomous_session_progress_fields(self, status: dict[str, Any]) -> None:
        if not isinstance(status, dict):
            return
        self._update_autonomous_session_health_fields(status)
        goal = dict(status.get("session_goal", {}) or {})
        context = dict(status.get("session_context", {}) or {})
        health = dict(status.get("session_health", {}) or {})
        lifecycle = dict(status.get("lifecycle", {}) or {})
        max_ticks = int(status.get("max_ticks", 0) or 0)
        tick_done = int(status.get("tick_done", 0) or 0)
        current_status = str(status.get("status", "") or "").strip() or "queued"
        phase_status_map = {
            "queued": "queued",
            "running": "active",
            "recovering": "active",
            "pausing": "active",
            "paused": "paused",
            "stopping": "stopping",
            "completed": "completed",
            "stopped": "stopped",
            "failed": "failed",
            "interrupted": "interrupted",
        }
        ratio = 0.0
        remaining = 0
        if max_ticks > 0:
            ratio = min(1.0, max(0.0, float(tick_done) / float(max_ticks)))
            remaining = max(0, max_ticks - tick_done)
        goal["target_kind"] = "bounded_ticks" if max_ticks > 0 else "open_ended"
        goal["target_tick_count"] = max_ticks
        goal["ticks_completed"] = tick_done
        goal["remaining_tick_budget"] = remaining
        goal["completion_ratio"] = round(ratio, 4)
        goal["phase_status"] = phase_status_map.get(current_status, current_status)
        goal["updated_at_ms"] = int(status.get("updated_at_ms", now_ms()) or now_ms())
        next_phase_label = self._infer_autonomous_session_phase_label(status)
        previous_phase_label = str(goal.get("phase_label", "") or "")
        if not previous_phase_label:
            goal["phase_label"] = next_phase_label
            goal["phase_index"] = int(goal.get("phase_index", 0) or 0)
        elif previous_phase_label != next_phase_label:
            goal["phase_label"] = next_phase_label
            goal["phase_index"] = int(goal.get("phase_index", 0) or 0) + 1
        else:
            goal["phase_label"] = next_phase_label
        if "phase_index" not in goal or goal.get("phase_index") is None:
            goal["phase_index"] = max(0, int(lifecycle.get("transition_count", 1) or 1) - 1)
        if not str(goal.get("goal_text", "") or "").strip():
            goal["goal_text"] = str(status.get("text_hint", "") or goal.get("label", "Autonomous Session") or "Autonomous Session")
        goal["focus_preview"] = [str(item or "") for item in (context.get("last_focus_preview", []) or []) if str(item or "")][:6]
        goal["selected_action_names"] = [str(item or "") for item in (context.get("last_selected_action_names", []) or []) if str(item or "")][:4]
        goal["recover_hint"] = str(health.get("recover_hint", "") or "")
        status["session_goal"] = goal

    def _ensure_autonomous_session_status_defaults(self, status: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(status, dict):
            return {}
        payload = copy.deepcopy(status)
        created_at_ms = int(payload.get("created_at_ms", 0) or now_ms())
        label = str(payload.get("label", "") or payload.get("session_label", "") or "Autonomous Session")
        text_hint = str(payload.get("text_hint", "") or "")
        max_ticks = int(payload.get("max_ticks", 0) or 0)

        goal = payload.get("session_goal")
        if not isinstance(goal, dict):
            goal = self._build_autonomous_session_goal(label=label, text_hint=text_hint, max_ticks=max_ticks or None, now_ts=created_at_ms)
        else:
            goal = copy.deepcopy(goal)
            goal.setdefault("label", label)
            goal.setdefault("goal_text", text_hint or label)
            goal.setdefault("goal_source", "text_hint" if text_hint else "label")
            goal.setdefault("target_kind", "bounded_ticks" if max_ticks > 0 else "open_ended")
            goal.setdefault("target_tick_count", max_ticks)
            goal.setdefault("phase_label", "queued_wait")
            goal.setdefault("phase_index", 0)
            goal.setdefault("phase_status", "queued")
            goal.setdefault("ticks_completed", 0)
            goal.setdefault("remaining_tick_budget", max_ticks)
            goal.setdefault("completion_ratio", 0.0)
            goal.setdefault("focus_preview", [])
            goal.setdefault("selected_action_names", [])
            goal.setdefault("recover_hint", "")
            goal.setdefault("updated_at_ms", created_at_ms)

        lifecycle = payload.get("lifecycle")
        if not isinstance(lifecycle, dict):
            lifecycle = self._build_autonomous_session_lifecycle(created_at_ms=created_at_ms, initial_status=str(payload.get("status", "") or "queued"))
        else:
            lifecycle = copy.deepcopy(lifecycle)
            lifecycle.setdefault("start_count", 1)
            lifecycle.setdefault("pause_request_count", 0)
            lifecycle.setdefault("paused_count", 0)
            lifecycle.setdefault("resume_count", 0)
            lifecycle.setdefault("recover_count", 0)
            lifecycle.setdefault("stop_request_count", 0)
            lifecycle.setdefault("interrupt_count", 0)
            lifecycle.setdefault("completion_count", 0)
            lifecycle.setdefault("failure_count", 0)
            lifecycle.setdefault("transition_count", 1)
            lifecycle.setdefault("last_transition", str(payload.get("status", "") or "queued"))
            lifecycle.setdefault("last_transition_at_ms", created_at_ms)
            lifecycle.setdefault("last_transition_reason", "")
            lifecycle.setdefault("last_status", str(payload.get("status", "") or "queued"))

        context = payload.get("session_context")
        if not isinstance(context, dict):
            context = self._build_autonomous_session_context(now_ts=created_at_ms)
        else:
            context = copy.deepcopy(context)
            context.setdefault("last_tick_id", "")
            context.setdefault("last_input_preview", "")
            context.setdefault("last_focus_preview", [])
            context.setdefault("last_bn_ids", [])
            context.setdefault("last_selected_action_names", [])
            context.setdefault("last_selected_action_statuses", [])
            context.setdefault("last_teacher_mode", "")
            context.setdefault("last_external_teacher_mode", "")
            context.setdefault("updated_at_ms", created_at_ms)

        health = payload.get("session_health")
        if not isinstance(health, dict):
            health = self._build_autonomous_session_health(now_ts=created_at_ms)
        else:
            health = copy.deepcopy(health)
            health.setdefault("health_status", "queued")
            health.setdefault("health_reason", "queued_waiting_for_first_tick")
            health.setdefault("recover_hint", "")
            health.setdefault("idle_ticks", 0)
            health.setdefault("capture_failures", 0)
            health.setdefault("action_errors", 0)
            health.setdefault("last_logic_ms", 0.0)
            health.setdefault("last_sleep_ms", 0)
            health.setdefault("last_screen_capture_ok", False)
            health.setdefault("last_focus_preview", [])
            health.setdefault("last_input_preview", "")
            health.setdefault("last_selected_action_names", [])
            health.setdefault("last_selected_action_statuses", [])
            health.setdefault("last_bn_ids", [])
            health.setdefault("last_tick_generated_at_ms", 0)
            health.setdefault("last_checkpoint_at_ms", 0)
            health.setdefault("last_checkpoint_tick_done", 0)
            health.setdefault("updated_at_ms", created_at_ms)

        payload["label"] = label
        payload["session_goal"] = goal
        payload["lifecycle"] = lifecycle
        payload["session_context"] = context
        payload["session_health"] = health
        self._update_autonomous_session_progress_fields(payload)
        return payload

    def _mark_autonomous_session_transition(self, status: dict[str, Any], *, transition: str, reason: str = "") -> None:
        if not isinstance(status, dict):
            return
        payload = self._ensure_autonomous_session_status_defaults(status)
        lifecycle = dict(payload.get("lifecycle", {}) or {})
        counter_key_by_transition = {
            "pause_requested": "pause_request_count",
            "paused": "paused_count",
            "resumed": "resume_count",
            "recovered": "recover_count",
            "stop_requested": "stop_request_count",
            "interrupted": "interrupt_count",
            "completed": "completion_count",
            "failed": "failure_count",
        }
        lifecycle["transition_count"] = int(lifecycle.get("transition_count", 0) or 0) + 1
        lifecycle["last_transition"] = str(transition or "")
        lifecycle["last_transition_at_ms"] = now_ms()
        lifecycle["last_transition_reason"] = str(reason or "")
        lifecycle["last_status"] = str(payload.get("status", "") or "")
        counter_key = counter_key_by_transition.get(str(transition or ""))
        if counter_key:
            lifecycle[counter_key] = int(lifecycle.get(counter_key, 0) or 0) + 1
        payload["lifecycle"] = lifecycle
        self._update_autonomous_session_progress_fields(payload)
        status.clear()
        status.update(payload)

    def _repair_bootstrap_autonomous_session_statuses(self, *, limit: int = 64) -> None:
        if not self.layout.runs_root.exists():
            return
        run_dirs = sorted(
            [path for path in self.layout.runs_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(1, int(limit))]
        running_states = {"queued", "running", "paused", "pausing", "recovering", "stopping"}
        for run_dir in run_dirs:
            status_path = run_dir / "live" / "autonomous_session_status.json"
            if not status_path.exists():
                continue
            raw_status = read_json(status_path, default={})
            if not isinstance(raw_status, dict) or not raw_status:
                continue
            normalized = self._normalize_bootstrap_autonomous_session_status(raw_status)
            if normalized == raw_status:
                continue
            write_json(status_path, normalized)
            manifest_path = run_dir / "manifest.json"
            manifest = read_json(manifest_path, default={})
            if isinstance(manifest, dict) and manifest:
                normalized_status = str(normalized.get("status", "") or "")
                if normalized_status:
                    manifest["status"] = normalized_status
                    manifest["updated_at_ms"] = int(normalized.get("updated_at_ms", now_ms()) or now_ms())
                    if int(manifest.get("finished_at_ms", 0) or 0) <= 0 and int(normalized.get("finished_at_ms", 0) or 0) > 0:
                        manifest["finished_at_ms"] = int(normalized.get("finished_at_ms", 0) or 0)
                    write_json(manifest_path, manifest)
            raw_status_name = str(raw_status.get("status", "") or "").strip()
            if bool(raw_status.get("active", False)) and raw_status_name in running_states and str(normalized.get("status", "") or "") == "interrupted":
                append_jsonl(
                    run_dir / "system" / "events.jsonl",
                    {
                        "ts_ms": now_ms(),
                        "type": "session_interrupted_on_bootstrap_repair",
                        "session_id": str(normalized.get("session_id", "") or ""),
                        "run_id": str(normalized.get("run_id", "") or run_dir.name),
                        "reason": str(normalized.get("last_stop_reason", "") or "session_status_restored_without_live_thread"),
                    },
                )

    def _bootstrap_latest_live_from_disk(self) -> None:
        runs = list_runs(self.layout, limit=1)
        if not runs:
            return
        self._sync_latest_live_from_disk_if_idle(force=True)

    def _sync_latest_live_from_disk_if_idle(self, *, force: bool = False) -> None:
        thread = self._active_thread
        if not force and thread is not None and thread.is_alive():
            return
        runs = list_runs(self.layout, limit=1)
        if not runs:
            return
        latest_row = dict(runs[0] or {})
        latest_run_id = str(latest_row.get("run_id", "") or "")
        if not latest_run_id:
            return
        current_latest_run_id = str(self._latest_live.get("latest_run_id", "") or "")
        if not force and current_latest_run_id == latest_run_id and self._latest_live.get("status") not in {"running", "queued", "paused", "pausing", "recovering", "stopping"}:
            return
        latest_run_dir = self.layout.runs_root / latest_run_id
        latest = read_json(latest_run_dir / "live" / "latest.json", default=None)
        if isinstance(latest, dict):
            self._latest_live = latest
            self._restore_live_ring(latest.get("recent_ticks", []) or [])
        else:
            manifest = self.get_manifest(latest_run_id)
            fallback_status = str((latest_row.get("status", "") or manifest.get("status", "") or "idle")).strip() or "idle"
            self._latest_live = {
                "schema_id": "live_snapshot/v1",
                "status": fallback_status,
                "active_run_id": latest_run_id if fallback_status in {"running", "queued", "paused", "pausing", "recovering", "stopping", "interrupted"} else "",
                "latest_run_id": latest_run_id,
                "latest_tick": {
                    "tick_index": int(manifest.get("latest_tick_index", -1) or -1),
                    "tick_id": "",
                    "input_preview": "",
                    "a_focus_preview": [],
                    "logic_ms": 0.0,
                },
                "recent_ticks": [],
                "server_time_ms": now_ms(),
            }
        session_status = self._normalize_bootstrap_autonomous_session_status(read_autonomous_session_status(latest_run_dir))
        if session_status:
            self._autonomous_session_status = session_status

    def _restore_live_ring(self, rows: list[dict[str, Any]]) -> None:
        unique_rows: list[dict[str, Any]] = []
        seen_tick_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            tick_id = str(row.get("tick_id", "") or "")
            if tick_id and tick_id in seen_tick_ids:
                continue
            if tick_id:
                seen_tick_ids.add(tick_id)
            unique_rows.append(copy.deepcopy(row))
        self._live_ring.clear()
        for row in unique_rows[-self.config.live_ring_limit :]:
            self._live_ring.append(row)

    def config_public(self) -> dict[str, Any]:
        data = self.config.to_dict()
        data["repo_root"] = str(self.repo_root)
        data["outputs_root_resolved"] = str(self.layout.outputs_root)
        data["server_meta"] = self.service_meta()
        return data

    def service_meta(self) -> dict[str, Any]:
        return {
            "service": "observatory_v2",
            "process_id": int(os.getpid()),
            "boot_id": str(self._service_boot_id),
            "started_at_ms": int(self._service_started_at_ms),
            "repo_root": str(self.repo_root),
            "outputs_root": str(self.layout.outputs_root),
        }

    def _ensure_runtime_mutation_idle(self, operation: str) -> None:
        thread = self._active_thread
        if thread is not None and thread.is_alive():
            raise AppError(f"{operation} 需要在当前 run / session 完成后执行。")

    def list_run_infos(self, limit: int = 32) -> list[dict[str, Any]]:
        return list_runs(self.layout, limit=limit)

    def latest_run_id(self) -> str:
        runs = self.list_run_infos(limit=1)
        if not runs:
            return ""
        return str(runs[0]["run_id"] or "")

    def _read_manifest_raw(self, run_id: str) -> dict[str, Any]:
        run_dir = self.layout.runs_root / run_id
        payload = read_json(run_dir / "manifest.json", default={})
        return payload if isinstance(payload, dict) else {}

    def get_manifest(self, run_id: str) -> dict[str, Any]:
        payload = self._read_manifest_raw(run_id)
        if not payload:
            return {}
        run_dir = self.layout.runs_root / run_id
        session_status = self._read_session_status_for_manifest_overlay(run_id=run_id, run_dir=run_dir)
        if session_status:
            return overlay_manifest_with_session_status(payload, session_status)
        return payload

    def get_live_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._sync_latest_live_from_disk_if_idle()
            live = copy.deepcopy(self._latest_live)
            live["recent_ticks"] = list(self._live_ring)
            live["server_time_ms"] = now_ms()
            live["known_runs"] = self.list_run_infos(limit=8)
            if self._autonomous_session_status:
                live["autonomous_session"] = copy.deepcopy(self._autonomous_session_status)
            if self._active_stream_source is not None:
                live["active_stream_source"] = self._active_stream_source.status()
            return live

    def get_tick_summary(self, run_id: str, tick_index: int) -> dict[str, Any]:
        run_dir = self.layout.runs_root / run_id
        path = chunk_file(run_dir, kind="summary", tick_index=tick_index, chunk_size=self.config.run_chunk_size)
        if not path.exists():
            return {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and int(row.get("tick_index", -1)) == int(tick_index):
                return row
        return {}

    def get_tick_sidecar(self, run_id: str, tick_index: int) -> dict[str, Any]:
        run_dir = self.layout.runs_root / run_id
        path = chunk_file(run_dir, kind="sidecar", tick_index=tick_index, chunk_size=self.config.run_chunk_size)
        if not path.exists():
            return {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and int(row.get("tick_index", -1)) == int(tick_index):
                return self._restore_externalized_sidecar_payloads(run_dir=run_dir, tick_index=tick_index, sidecar=row)
        return {}

    def get_run_events(self, run_id: str, *, limit: int = 120) -> list[dict[str, Any]]:
        run_dir = self.layout.runs_root / run_id
        path = run_dir / "system" / "events.jsonl"
        if not path.exists():
            return []
        rows = list(iter_jsonl(path))
        if not rows:
            return []
        capped_limit = max(1, int(limit or 120))
        return rows[-capped_limit:]

    def list_tick_summaries(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        run_dir = self.layout.runs_root / run_id
        cached = read_json(run_dir / "live" / "tick_list.json", default=None)
        max_rows = int(limit or self.config.observatory_tick_list_limit)
        if isinstance(cached, dict) and isinstance(cached.get("ticks"), list):
            rows = [dict(item) for item in cached.get("ticks", []) if isinstance(item, dict)]
            rows.sort(key=lambda item: int(-1 if item.get("tick_index", -1) is None else item.get("tick_index", -1)))
            return rows[-max_rows:]
        chunks = sorted((run_dir / "chunks").glob("*.summary.jsonl"))
        rows: list[dict[str, Any]] = []
        for path in chunks:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        rows.sort(key=lambda item: int(-1 if item.get("tick_index", -1) is None else item.get("tick_index", -1)))
        return rows[-max_rows:]

    def build_run_overview(self, run_id: str) -> dict[str, Any]:
        manifest = self.get_manifest(run_id)
        if not manifest:
            return {}
        rollup = self.get_run_rollup(run_id)
        if rollup:
            latest_tick = dict(rollup.get("last_summary", {}) or {})
            return {
                "run_id": run_id,
                "manifest": manifest,
                "tick_count": int(rollup.get("tick_count", 0) or 0),
                "mean_logic_ms": float((rollup.get("logic_ms", {}) or {}).get("mean", 0.0) or 0.0),
                "max_logic_ms": float((rollup.get("logic_ms", {}) or {}).get("max", 0.0) or 0.0),
                "focus_preview": list(rollup.get("focus_preview_tail", []) or []),
                "input_preview": list(rollup.get("input_preview_tail", []) or []),
                "bn_count_series_tail": list(((rollup.get("series_tail", {}) or {}).get("bn_count", []) or [])),
                "memory_index_summary": (rollup.get("last_summary", {}) or {}).get("memory_index_summary", self._runtime.memory_store.index_summary()),
                "executor_status": self._agent_sandbox.status(),
                "executor_recent_events": self._agent_sandbox.recent_events(limit=8),
                "latest_stream_source": ((latest_tick.get("multimodal_summary", {}) or {}).get("stream_source", {}) or {}),
                "latest_teacher_review": ((latest_tick.get("rules_preview", {}) or {}).get("teacher_review", {}) or {}),
                "latest_teacher_feedback": ((latest_tick.get("rules_preview", {}) or {}).get("teacher_feedback", {}) or {}),
                "rollup": rollup,
            }
        ticks = self.list_tick_summaries(run_id, limit=self.config.observatory_tick_list_limit)
        logic_values = []
        focus_texts = []
        bn_counts = []
        input_previews = []
        for row in ticks:
            preview = str((row.get("input_preview", "") or ""))[:96]
            if preview:
                input_previews.append(preview)
            focus = " ".join(str(item or "") for item in (row.get("a_focus_preview", []) or []) if str(item or ""))
            if focus:
                focus_texts.append(focus)
            bn_counts.append(len(row.get("bn_preview", []) or []))
        metric_chunks = sorted((self.layout.runs_root / run_id / "chunks").glob("*.metrics.jsonl"))
        for path in metric_chunks:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    logic_values.append(float(payload.get("logic_ms", 0.0) or 0.0))
        return {
            "run_id": run_id,
            "manifest": manifest,
            "tick_count": len(ticks),
            "mean_logic_ms": round(sum(logic_values) / len(logic_values), 4) if logic_values else 0.0,
            "max_logic_ms": round(max(logic_values), 4) if logic_values else 0.0,
            "focus_preview": focus_texts[-8:],
            "input_preview": input_previews[-8:],
            "bn_count_series_tail": bn_counts[-32:],
            "memory_index_summary": self._runtime.memory_store.index_summary(),
            "executor_status": self._agent_sandbox.status(),
            "executor_recent_events": self._agent_sandbox.recent_events(limit=8),
            "latest_stream_source": ((ticks[-1].get("multimodal_summary", {}) or {}).get("stream_source", {}) or {}) if ticks else {},
            "latest_teacher_review": ((ticks[-1].get("rules_preview", {}) or {}).get("teacher_review", {}) or {}) if ticks else {},
            "latest_teacher_feedback": ((ticks[-1].get("rules_preview", {}) or {}).get("teacher_feedback", {}) or {}) if ticks else {},
        }

    def build_run_overview_batch(self, *, limit: int = 8, run_ids: list[str] | None = None) -> dict[str, Any]:
        capped_limit = max(1, min(int(limit or 8), 32))
        infos = self.list_run_infos(limit=max(capped_limit, 32 if run_ids else capped_limit))
        info_by_id = {str(item.get("run_id", "") or ""): dict(item) for item in infos if str(item.get("run_id", "") or "")}
        ordered_ids = [str(item.get("run_id", "") or "") for item in infos[:capped_limit]]
        if run_ids:
            ordered_ids = []
            seen: set[str] = set()
            for raw in run_ids:
                clean = str(raw or "").strip()
                if not clean or clean in seen:
                    continue
                ordered_ids.append(clean)
                seen.add(clean)
            ordered_ids = ordered_ids[:capped_limit]

        rows: list[dict[str, Any]] = []
        for run_id in ordered_ids:
            overview = self.build_run_overview(run_id)
            if overview:
                base = info_by_id.get(run_id, {})
                rows.append({**base, **overview, "__hydrated": True})
                continue
            if run_id in info_by_id:
                rows.append({**info_by_id[run_id], "run_id": run_id, "__hydrated": False})
        return {
            "schema_id": "run_overview_batch/v1",
            "schema_version": "1.0",
            "limit": capped_limit,
            "count": len(rows),
            "runs": rows,
        }

    def get_run_rollup(self, run_id: str) -> dict[str, Any]:
        run_dir = self.layout.runs_root / run_id
        payload = read_json(run_dir / "live" / "run_rollup.json", default={})
        if isinstance(payload, dict):
            return payload
        return {}

    def get_rules_payload(self) -> dict[str, Any]:
        return self._runtime.rules_engine.export_rules()

    def save_rules_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.rules_engine.save_rules(payload)

    def validate_rules_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.rules_engine.validate_rules(payload)

    def simulate_rules(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        context = dict(payload or {})
        rules_payload = context.pop("rules_payload", None)
        tuner_payload = context.pop("tuner_payload", None)
        if not context:
            context = {
                "tick_index": 0,
                "state_top": self._runtime.state_pool.snapshot_top(limit=10),
                "state_pool_summary": self._runtime.state_pool.snapshot_summary(),
                "bn_list": [],
                "c_star": {"items": []},
                "runtime_metrics": {"logic_ms": 0.0},
            }
        return self._runtime.rules_engine.simulate(
            context,
            rules_payload=rules_payload if isinstance(rules_payload, dict) else None,
            tuner_payload=tuner_payload if isinstance(tuner_payload, dict) else None,
        )

    def get_tuner_payload(self) -> dict[str, Any]:
        return self._runtime.rules_engine.export_tuner()

    def save_tuner_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.rules_engine.save_tuner(payload)

    def validate_tuner_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.rules_engine.validate_tuner(payload)

    def executor_status(self) -> dict[str, Any]:
        return {
            **self._agent_sandbox.status(),
            "recent_events": self._agent_sandbox.recent_events(limit=16),
            "action_learning_bias_summary": self._runtime.action_learning.bias_summary(limit=12),
            "action_learning_recent_feedback": self._runtime.action_learning.recent_feedback(limit=16),
            "tuner_learning_target_bias_summary": self._runtime.tuner_learning.target_bias_summary(limit=12),
            "tuner_learning_profile_bias_summary": self._runtime.tuner_learning.profile_bias_summary(limit=12),
            "tuner_learning_recent_feedback": self._runtime.tuner_learning.recent_feedback(limit=16),
            "teacher_layer": self._runtime.teacher_layer.export_payload(),
        }

    def export_runtime_summary(self) -> dict[str, Any]:
        state_summary = self._runtime.state_pool.snapshot_summary()
        short_term_snapshot = self._runtime.short_term.snapshot()
        autonomous_session_status = self.get_autonomous_session_status()
        return {
            "schema_id": "runtime_summary/v1",
            "schema_version": "1.0",
            "export_meta": {
                "memory_count": self._runtime.memory_store.count(),
                "latest_run_id": self.latest_run_id(),
                "exported_at_ms": now_ms(),
                "memory_index_summary": self._runtime.memory_store.index_summary(),
                "executor_status": self._agent_sandbox.status(),
                "action_learning_bias_summary": self._runtime.action_learning.bias_summary(limit=12),
                "tuner_learning_target_bias_summary": self._runtime.tuner_learning.target_bias_summary(limit=12),
                "tuner_learning_profile_bias_summary": self._runtime.tuner_learning.profile_bias_summary(limit=12),
                "teacher_layer": self._runtime.teacher_layer.export_payload(),
                "last_forget_summary": copy.deepcopy(self._last_forget_summary),
                "last_forget_preview_summary": copy.deepcopy(self._last_forget_preview_summary),
                "autonomous_session_status": autonomous_session_status,
            },
            "state_pool_summary": {
                "tick_index": state_summary.get("tick_index", -1),
                "state_pool_size": state_summary.get("state_pool_size", 0),
                "recent_external_count": state_summary.get("recent_external_count", 0),
                "verbatim_chars": state_summary.get("verbatim_chars", 0),
                "anchor_summary": state_summary.get("anchor_summary", {}),
                "residual_summary": state_summary.get("residual_summary", {}),
                "handle_summary": state_summary.get("handle_summary", {}),
                "recent_external_summary": state_summary.get("recent_external_summary", []),
            },
            "short_term_summary": {
                "count": len(short_term_snapshot),
                "tail": short_term_snapshot[-8:],
            },
            "sa_registry_summary": {
                "prototype_count": len(self._runtime.sa_registry.all_prototypes()),
            },
            "action_learning_summary": {
                "bias_count": len(self._runtime.action_learning.bias_summary(limit=256)),
                "bias_top": self._runtime.action_learning.bias_summary(limit=12),
                "recent_feedback": self._runtime.action_learning.recent_feedback(limit=16),
            },
            "tuner_learning_summary": {
                "target_bias_count": len(self._runtime.tuner_learning.target_bias_summary(limit=256)),
                "target_bias_top": self._runtime.tuner_learning.target_bias_summary(limit=12),
                "profile_bias_top": self._runtime.tuner_learning.profile_bias_summary(limit=12),
                "recent_feedback": self._runtime.tuner_learning.recent_feedback(limit=16),
            },
            "teacher_layer_summary": self._runtime.teacher_layer.export_payload(),
            "autonomous_session_summary": autonomous_session_status,
        }

    def capture_screen_preview(self) -> dict[str, Any]:
        payload = self._agent_sandbox.capture_screenshot_packet(force=True)
        safe_payload = dict(payload)
        if "image_bytes" in safe_payload:
            safe_payload["image_bytes_len"] = len(bytes(safe_payload.get("image_bytes") or b""))
            safe_payload.pop("image_bytes", None)
        return safe_payload

    def execute_manual_action(self, *, action_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._agent_sandbox.execute_manual_action(action_name=action_name, params=params)

    def export_runtime(self) -> dict[str, Any]:
        return {
            "schema_id": "runtime_export/v1",
            "schema_version": "1.1",
            "runtime": self._runtime.export_payload(),
            "sandbox": self._agent_sandbox.export_payload(),
            "export_meta": {
                "memory_count": self._runtime.memory_store.count(),
                "latest_run_id": self.latest_run_id(),
                "exported_at_ms": now_ms(),
                "memory_index_summary": self._runtime.memory_store.index_summary(),
                "executor_status": self._agent_sandbox.status(),
                "action_learning_bias_summary": self._runtime.action_learning.bias_summary(limit=12),
                "tuner_learning_target_bias_summary": self._runtime.tuner_learning.target_bias_summary(limit=12),
                "tuner_learning_profile_bias_summary": self._runtime.tuner_learning.profile_bias_summary(limit=12),
                "teacher_layer": self._runtime.teacher_layer.export_payload(),
                "last_forget_summary": copy.deepcopy(self._last_forget_summary),
                "last_forget_preview_summary": copy.deepcopy(self._last_forget_preview_summary),
            },
        }

    def import_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ensure_runtime_mutation_idle("runtime 导入")
        runtime_payload = payload.get("runtime", payload)
        if not isinstance(runtime_payload, dict):
            raise AppError("runtime payload 无效")
        self._runtime.import_payload(runtime_payload)
        sandbox_payload = payload.get("sandbox", {})
        if isinstance(sandbox_payload, dict):
            self._agent_sandbox.import_payload(sandbox_payload)
        export_meta = payload.get("export_meta", {})
        if isinstance(export_meta, dict) and ("last_forget_summary" in export_meta or "last_forget_preview_summary" in export_meta):
            last_forget_summary = export_meta.get("last_forget_summary", {})
            last_forget_preview_summary = export_meta.get("last_forget_preview_summary", {})
            with self._lock:
                self._last_forget_summary = (
                    copy.deepcopy(last_forget_summary) if isinstance(last_forget_summary, dict) else {}
                )
                self._last_forget_preview_summary = (
                    copy.deepcopy(last_forget_preview_summary) if isinstance(last_forget_preview_summary, dict) else {}
                )
                self._persist_service_runtime_state()
        return {"ok": True, "memory_count": self._runtime.memory_store.count()}

    def save_checkpoint(self, path: Path) -> dict[str, Any]:
        payload = self.export_runtime()
        write_json(path, payload)
        return {"ok": True, "path": str(path)}

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        payload = read_json(path, default={})
        if not isinstance(payload, dict):
            raise AppError("checkpoint 内容无效")
        return self.import_runtime(payload)

    def forget_cold_memories(
        self,
        *,
        keep_latest: int,
        min_reality_weight: float = 0.0,
        min_total_item_energy: float = 0.0,
        protect_memory_kinds: list[str] | None = None,
        max_memory_count: int | None = None,
        strategy: str = "latest_only",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_runtime_mutation_idle("memory forget")
            result = self._runtime.memory_store.forget_cold_memories(
                keep_latest=keep_latest,
                min_reality_weight=min_reality_weight,
                min_total_item_energy=min_total_item_energy,
                protect_memory_kinds=protect_memory_kinds,
                max_memory_count=max_memory_count,
                strategy=strategy,
                dry_run=dry_run,
            )
            summary = {
                **dict(result or {}),
                "generated_at_ms": now_ms(),
            }
            if dry_run:
                self._last_forget_preview_summary = summary
            else:
                self._last_forget_summary = summary
            self._persist_service_runtime_state()
            return dict(summary)

    def export_memory_deployment_bundle(self, directory: Path) -> dict[str, Any]:
        return self._runtime.memory_store.save_deployment_bundle(directory)

    def import_memory_deployment_bundle(self, directory: Path) -> dict[str, Any]:
        with self._lock:
            self._ensure_runtime_mutation_idle("memory bundle 导入")
        return self._runtime.memory_store.load_deployment_bundle(directory)

    def inspect_memory_deployment_bundle(self, directory: Path) -> dict[str, Any]:
        return self._runtime.memory_store.inspect_deployment_bundle(directory)

    def start_demo_run(
        self,
        *,
        tick_count: int | None = None,
        tick_interval_ms: int | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise AppError("已有运行中的 demo run，请等待其完成后再启动。")
            planned = int(tick_count or self.config.default_demo_tick_count)
            interval = int(tick_interval_ms or self.config.default_demo_tick_interval_ms)
            run_id = make_run_id("phase1_demo")
            run_dir = self._prepare_run_dir(run_id)
            manifest = self._build_manifest(run_id=run_id, run_dir=run_dir, tick_planned=planned, label=label or "Phase1 最小演示运行")
            self._write_manifest(run_dir, manifest)
            append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": now_ms(), "type": "run_created", "run_id": run_id})
            self._active_run_id = run_id
            self._active_thread = threading.Thread(
                target=self._run_demo_loop,
                kwargs={"run_id": run_id, "run_dir": run_dir, "tick_count": planned, "tick_interval_ms": interval},
                daemon=True,
                name=f"demo-run-{run_id}",
            )
            self._active_thread.start()
            return {"run_id": run_id, "run_dir": str(run_dir), "status": "queued"}

    def start_text_run(
        self,
        *,
        texts: list[str],
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
    ) -> dict[str, Any]:
        clean_texts = [str(item or "") for item in texts]
        return self.start_multimodal_run(
            items=[{"text": text, "source_type": "external_text"} for text in clean_texts],
            label=label or "Phase2 文本最小闭环运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase2_text_min_loop",
            notes=[
                "这是 AP 二期 Phase 2 的文本输入最小闭环运行。",
                "当前已接入文本感受器最小实现、状态池最小内核、R_state、Bn、C*、规则层和短期记忆。",
            ],
        )

    def start_multimodal_run(
        self,
        *,
        items: list[dict[str, Any]],
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        run_kind: str = "phase11_multimodal_run",
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_items = [dict(item) for item in items]
        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise AppError("已有运行中的 run，请等待其完成后再启动。")
            run_id = make_run_id(run_kind)
            run_dir = self._prepare_run_dir(run_id)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(clean_items):
                append_jsonl(
                    run_dir / "inputs" / "inputs.jsonl",
                    {
                        "schema_id": "multimodal_input_envelope/v1",
                        "schema_version": "1.0",
                        "tick_index": index,
                        **self._serialize_input_item(item),
                    },
                )
            manifest = self._build_manifest(
                run_id=run_id,
                run_dir=run_dir,
                tick_planned=len(clean_items),
                label=label or "Phase11 多模态统一运行",
                run_kind=run_kind,
                notes=notes
                or [
                    "这是 AP 二期多模态统一编排运行。",
                    "同一条主链现在可以同时接入文本、图片、音频，并在统一 sidecar 中观测。",
                ],
            )
            self._write_manifest(run_dir, manifest)
            append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": now_ms(), "type": "run_created", "run_id": run_id})
            if reset_runtime:
                self._runtime = self._build_runtime()
                self._agent_sandbox = self._build_agent_sandbox()
            self._active_run_id = run_id
            self._active_thread = threading.Thread(
                target=self._run_multimodal_loop,
                kwargs={
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "items": clean_items,
                    "tick_interval_ms": int(tick_interval_ms),
                    "base_tick_index": self._runtime.memory_store.latest_tick_index() + 1,
                },
                daemon=True,
                name=f"multimodal-run-{run_id}",
            )
            self._active_thread.start()
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "queued"}

    def start_realtime_source_run(
        self,
        *,
        source: BaseRealtimeSourceV1,
        text_prefix: str = "",
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        run_kind: str = "phase18_realtime_source_run",
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                try:
                    source.close()
                except Exception:
                    pass
                raise AppError("已有运行中的 run，请等待其完成后再启动。")
            run_id = make_run_id(run_kind)
            run_dir = self._prepare_run_dir(run_id)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            manifest = self._build_manifest(
                run_id=run_id,
                run_dir=run_dir,
                tick_planned=max(0, int(source.total_items or 0)),
                label=label or "Phase18 Realtime Source Run",
                run_kind=run_kind,
                notes=notes
                or [
                    "这是 AP 二期统一 realtime source 运行。",
                    "输入源按 tick 懒加载消费，避免先把整条流全部展开到内存。",
                ],
            )
            manifest["source_mode"] = "realtime_source"
            manifest["source_meta"] = source.status()
            self._write_manifest(run_dir, manifest)
            append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": now_ms(), "type": "run_created", "run_id": run_id})
            if reset_runtime:
                self._runtime = self._build_runtime()
                self._agent_sandbox = self._build_agent_sandbox()
            self._active_run_id = run_id
            self._active_stream_source = source
            self._active_thread = threading.Thread(
                target=self._run_realtime_source_loop,
                kwargs={
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "source": source,
                    "text_prefix": str(text_prefix or ""),
                    "tick_interval_ms": int(tick_interval_ms),
                    "base_tick_index": self._runtime.memory_store.latest_tick_index() + 1,
                },
                daemon=True,
                name=f"realtime-source-run-{run_id}",
            )
            self._active_thread.start()
            return {"run_id": run_id, "run_dir": str(run_dir), "status": "queued"}

    def continue_from_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        texts: list[str],
        label: str | None = None,
        tick_interval_ms: int = 0,
    ) -> dict[str, Any]:
        self.load_checkpoint(checkpoint_path)
        return self.start_text_run(
            texts=texts,
            label=label or "从 checkpoint 继续的文本运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=False,
        )

    def start_audio_stream_run(
        self,
        *,
        audio_bytes: bytes,
        text_prefix: str = "",
        tick_window_ms: int | None = None,
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
    ) -> dict[str, Any]:
        window_ms = int(tick_window_ms or self.config.hearing_window_ms)
        audio_source = self._stream_adapter.build_audio_file_source(raw_bytes=audio_bytes, tick_window_ms=window_ms, source_type="audio_stream")
        return self.start_realtime_source_run(
            source=audio_source,
            text_prefix=text_prefix,
            label=label or "Phase12 连续音频流运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase12_audio_stream_run",
            notes=[
                "这是 AP 二期连续音频流适配运行。",
                "长音频先按固定 tick 窗口切分，再复用现有听觉感受器与主链。",
            ],
        )

    def start_video_stream_run(
        self,
        *,
        video_bytes: bytes,
        video_name: str = "",
        text_prefix: str = "",
        tick_fps: float | None = None,
        frame_stride: int | None = None,
        max_frames: int | None = None,
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
    ) -> dict[str, Any]:
        video_source = self._stream_adapter.build_video_file_source(
            raw_bytes=video_bytes,
            tick_fps=tick_fps,
            frame_stride=frame_stride,
            max_frames=max_frames,
            source_type="video_stream",
            file_hint=video_name,
        )
        if video_source.unavailable:
            raise AppError(video_source.last_error or "video stream 未能初始化")
        return self.start_realtime_source_run(
            source=video_source,
            text_prefix=text_prefix,
            label=label or "Phase20 连续视频流运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase20_video_stream_run",
            notes=[
                "这是 AP 二期连续视频流适配运行。",
                "视频先按固定 stride 或 tick_fps 解成逐 tick 帧，再复用现有视觉感受器与主链。",
            ],
        )

    def start_webcam_stream_run(
        self,
        *,
        text_prefix: str = "",
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        max_frames: int | None = None,
        device_index: int = 0,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> dict[str, Any]:
        webcam_source = self._stream_adapter.build_webcam_source(
            source_type="webcam_stream",
            device_index=device_index,
            max_frames=max_frames,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if webcam_source.unavailable:
            raise AppError(webcam_source.last_error or "webcam source unavailable")
        return self.start_realtime_source_run(
            source=webcam_source,
            text_prefix=text_prefix,
            label=label or "Phase21 Webcam Stream Run",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase21_webcam_stream_run",
            notes=[
                "这是 AP 二期 webcam 实时流运行。",
                "摄像头逐帧作为 realtime source 按 tick 懒加载进入视觉主链。",
            ],
        )

    def start_microphone_stream_run(
        self,
        *,
        text_prefix: str = "",
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        tick_window_ms: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        max_windows: int | None = None,
    ) -> dict[str, Any]:
        microphone_source = self._stream_adapter.build_microphone_source(
            source_type="microphone_stream",
            tick_window_ms=int(tick_window_ms or self.config.hearing_window_ms),
            sample_rate=sample_rate,
            channels=channels,
            device_index=device_index,
            max_windows=max_windows,
        )
        if microphone_source.unavailable:
            raise AppError(microphone_source.last_error or "microphone source unavailable")
        return self.start_realtime_source_run(
            source=microphone_source,
            text_prefix=text_prefix,
            label=label or "Phase21 Microphone Stream Run",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase21_microphone_stream_run",
            notes=[
                "这是 AP 二期 microphone 实时流运行。",
                "麦克风按固定窗口采样，作为 realtime source 按 tick 懒加载进入听觉主链。",
            ],
        )

    def start_autonomous_run(
        self,
        *,
        ticks: int,
        text_hint: str = "",
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        label: str | None = None,
        reward_schedule: list[dict[str, Any]] | None = None,
        stop_on_capture_failures: int | None = None,
        stop_on_action_errors: int | None = None,
        stop_on_idle_ticks: int | None = None,
        idle_backoff_ms: int | None = None,
        auto_feedback_enabled: bool | None = None,
        teacher_mode: str | None = None,
        llm_gate_mode: str | None = None,
        external_teacher_enabled: bool | None = None,
        external_teacher_mode: str | None = None,
        external_teacher_stub_response_path: str | None = None,
        external_teacher_fail_open: bool | None = None,
        external_teacher_max_retries: int | None = None,
        external_teacher_retry_backoff_ms: int | None = None,
        external_teacher_http_endpoint: str | None = None,
        external_teacher_http_headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        planned = max(1, int(ticks))
        resolved_teacher_mode = str(teacher_mode or self.config.autonomous_teacher_mode)
        resolved_external_teacher_enabled = (
            bool(external_teacher_enabled)
            if external_teacher_enabled is not None
            else bool(self.config.autonomous_external_teacher_enabled or resolved_teacher_mode == "llm_assisted")
        )
        resolved_external_teacher_mode = str(
            external_teacher_mode
            or self.config.autonomous_external_teacher_mode
            or ("stub_file" if resolved_teacher_mode == "llm_assisted" else "off")
        )
        if resolved_teacher_mode == "llm_assisted" and resolved_external_teacher_mode == "off":
            resolved_external_teacher_mode = "stub_file"
        items: list[dict[str, Any]] = []
        reward_by_tick: dict[int, dict[str, Any]] = {}
        for raw in reward_schedule or []:
            if not isinstance(raw, dict):
                continue
            reward_by_tick[int(raw.get("tick_index", -1) or -1)] = {
                "reward": float(raw.get("reward", 0.0) or 0.0),
                "punishment": float(raw.get("punishment", 0.0) or 0.0),
            }
        for index in range(planned):
            item: dict[str, Any] = {
                "text": str(text_hint or ""),
                "source_type": "autonomous_loop",
                "capture_screen": True,
                "autonomous_tick_meta": {
                    "planned_tick_index": index,
                    "stop_on_capture_failures": int(stop_on_capture_failures or self.config.autonomous_stop_on_consecutive_capture_failures),
                    "stop_on_action_errors": int(stop_on_action_errors or self.config.autonomous_stop_on_consecutive_action_errors),
                    "stop_on_idle_ticks": int(stop_on_idle_ticks or self.config.autonomous_stop_on_consecutive_idle_ticks),
                    "idle_backoff_ms": int(idle_backoff_ms or self.config.autonomous_idle_backoff_ms),
                    "auto_feedback_enabled": bool(self.config.autonomous_auto_feedback_enabled if auto_feedback_enabled is None else auto_feedback_enabled),
                    "teacher_mode": resolved_teacher_mode,
                    "llm_gate_mode": str(llm_gate_mode or self.config.autonomous_llm_gate_mode),
                    "external_teacher_enabled": resolved_external_teacher_enabled,
                    "external_teacher_mode": resolved_external_teacher_mode,
                    "external_teacher_stub_response_path": str(
                        external_teacher_stub_response_path
                        if external_teacher_stub_response_path is not None
                        else self.config.autonomous_external_teacher_stub_response_path
                    ),
                    "external_teacher_fail_open": (
                        bool(external_teacher_fail_open)
                        if external_teacher_fail_open is not None
                        else bool(self.config.autonomous_external_teacher_fail_open)
                    ),
                    "external_teacher_max_retries": int(
                        external_teacher_max_retries
                        if external_teacher_max_retries is not None
                        else self.config.autonomous_external_teacher_max_retries
                    ),
                    "external_teacher_retry_backoff_ms": int(
                        external_teacher_retry_backoff_ms
                        if external_teacher_retry_backoff_ms is not None
                        else self.config.autonomous_external_teacher_retry_backoff_ms
                    ),
                    "external_teacher_http_endpoint": str(
                        external_teacher_http_endpoint
                        if external_teacher_http_endpoint is not None
                        else self.config.autonomous_external_teacher_http_endpoint
                    ),
                    "external_teacher_http_headers": dict(
                        external_teacher_http_headers
                        if external_teacher_http_headers is not None
                        else self.config.autonomous_external_teacher_http_headers
                    ),
                },
            }
            feedback = reward_by_tick.get(index)
            if feedback:
                item["external_feedback"] = feedback
            items.append(item)
        return self.start_multimodal_run(
            items=items,
            label=label or "Phase19 自主电脑循环运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase19_autonomous_loop_run",
            notes=[
                "这是 AP 二期自主电脑循环运行。",
                "每个 tick 自动截图进入主链，并统一执行动作、记录反馈与日志。",
                "当前已加入长期运行所需的自动反馈、容错阈值与提前停止条件。",
            ],
        )

    def get_autonomous_session_status(self) -> dict[str, Any]:
        with self._lock:
            if not self._autonomous_session_status:
                return {"active": False, "session_id": "", "status": "idle"}
            return self._ensure_autonomous_session_status_defaults(copy.deepcopy(self._autonomous_session_status))

    def _read_session_status_for_manifest_overlay(self, *, run_id: str, run_dir: Path) -> dict[str, Any]:
        with self._lock:
            thread = self._active_thread
            live_status = self._autonomous_session_status or {}
            live_run_id = str(live_status.get("run_id", "") or "")
            if (
                run_id
                and run_id == str(self._active_run_id or "")
                and live_run_id == run_id
                and thread is not None
                and thread.is_alive()
            ):
                return self._ensure_autonomous_session_status_defaults(copy.deepcopy(live_status))
        session_status = read_autonomous_session_status(run_dir)
        if not session_status:
            return {}
        return self._normalize_bootstrap_autonomous_session_status(session_status)

    def start_autonomous_session(
        self,
        *,
        text_hint: str = "",
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
        label: str | None = None,
        max_ticks: int | None = None,
        stop_on_capture_failures: int | None = None,
        stop_on_action_errors: int | None = None,
        stop_on_idle_ticks: int | None = None,
        idle_backoff_ms: int | None = None,
        auto_feedback_enabled: bool | None = None,
        teacher_mode: str | None = None,
        llm_gate_mode: str | None = None,
        external_teacher_enabled: bool | None = None,
        external_teacher_mode: str | None = None,
        external_teacher_stub_response_path: str | None = None,
        external_teacher_fail_open: bool | None = None,
        external_teacher_max_retries: int | None = None,
        external_teacher_retry_backoff_ms: int | None = None,
        external_teacher_http_endpoint: str | None = None,
        external_teacher_http_headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise AppError("已有运行中的 run 或 autonomous session，请等待其完成后再启动。")
            run_id = make_run_id("phase20_autonomous_session")
            run_dir = self._prepare_run_dir(run_id)
            session_id = f"session::{run_id}"
            manifest = self._build_manifest(
                run_id=run_id,
                run_dir=run_dir,
                tick_planned=max(0, int(max_ticks or 0)),
                label=label or "Phase20 持续自主 session",
                run_kind="phase20_autonomous_session_run",
                notes=[
                    "这是 AP 二期持续自主 session。",
                    "它与 batch autonomous run 共用同一条主链，但改为逐 tick 动态生成输入。",
                    "支持 start / pause / resume / stop / status，用于更长期的自主闭环验证。",
                ],
            )
            manifest["session_mode"] = "continuous_autonomous"
            manifest["session_id"] = session_id
            self._write_manifest(run_dir, manifest)
            append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": now_ms(), "type": "run_created", "run_id": run_id})
            if reset_runtime:
                self._runtime = self._build_runtime()
                self._agent_sandbox = self._build_agent_sandbox()
            pause_event = threading.Event()
            stop_event = threading.Event()
            resolved_teacher_mode = str(teacher_mode or self.config.autonomous_teacher_mode)
            resolved_external_teacher_enabled = (
                bool(external_teacher_enabled)
                if external_teacher_enabled is not None
                else bool(self.config.autonomous_external_teacher_enabled or resolved_teacher_mode == "llm_assisted")
            )
            resolved_external_teacher_mode = str(
                external_teacher_mode
                or self.config.autonomous_external_teacher_mode
                or ("stub_file" if resolved_teacher_mode == "llm_assisted" else "off")
            )
            if resolved_teacher_mode == "llm_assisted" and resolved_external_teacher_mode == "off":
                resolved_external_teacher_mode = "stub_file"
            session_status = {
                "active": True,
                "session_id": session_id,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "label": label or "Phase20 持续自主 session",
                "status": "queued",
                "paused": False,
                "stopping": False,
                "tick_done": 0,
                "max_ticks": int(max_ticks or 0),
                "tick_interval_ms": int(tick_interval_ms),
                "text_hint": str(text_hint or ""),
                "created_at_ms": now_ms(),
                "started_at_ms": 0,
                "updated_at_ms": now_ms(),
                "finished_at_ms": 0,
                "base_tick_index": int(self._runtime.memory_store.latest_tick_index() + 1),
                "last_runtime_tick_index": int(self._runtime.memory_store.latest_tick_index()),
                "last_tick_index": -1,
                "last_stop_reason": "",
                "recoverable": True,
                "autonomous_tick_meta": {
                    "stop_on_capture_failures": int(stop_on_capture_failures or self.config.autonomous_stop_on_consecutive_capture_failures),
                    "stop_on_action_errors": int(stop_on_action_errors or self.config.autonomous_stop_on_consecutive_action_errors),
                    "stop_on_idle_ticks": int(stop_on_idle_ticks or self.config.autonomous_stop_on_consecutive_idle_ticks),
                    "idle_backoff_ms": int(idle_backoff_ms or self.config.autonomous_idle_backoff_ms),
                    "auto_feedback_enabled": bool(self.config.autonomous_auto_feedback_enabled if auto_feedback_enabled is None else auto_feedback_enabled),
                    "teacher_mode": resolved_teacher_mode,
                    "llm_gate_mode": str(llm_gate_mode or self.config.autonomous_llm_gate_mode),
                    "external_teacher_enabled": resolved_external_teacher_enabled,
                    "external_teacher_mode": resolved_external_teacher_mode,
                    "external_teacher_stub_response_path": str(
                        external_teacher_stub_response_path
                        if external_teacher_stub_response_path is not None
                        else self.config.autonomous_external_teacher_stub_response_path
                    ),
                    "external_teacher_fail_open": (
                        bool(external_teacher_fail_open)
                        if external_teacher_fail_open is not None
                        else bool(self.config.autonomous_external_teacher_fail_open)
                    ),
                    "external_teacher_max_retries": int(
                        external_teacher_max_retries
                        if external_teacher_max_retries is not None
                        else self.config.autonomous_external_teacher_max_retries
                    ),
                    "external_teacher_retry_backoff_ms": int(
                        external_teacher_retry_backoff_ms
                        if external_teacher_retry_backoff_ms is not None
                        else self.config.autonomous_external_teacher_retry_backoff_ms
                    ),
                    "external_teacher_http_endpoint": str(
                        external_teacher_http_endpoint
                        if external_teacher_http_endpoint is not None
                        else self.config.autonomous_external_teacher_http_endpoint
                    ),
                    "external_teacher_http_headers": dict(
                        external_teacher_http_headers
                        if external_teacher_http_headers is not None
                        else self.config.autonomous_external_teacher_http_headers
                    ),
                },
            }
            session_status = self._ensure_autonomous_session_status_defaults(session_status)
            self._active_run_id = run_id
            self._active_session_id = session_id
            self._autonomous_session_status = session_status
            self._autonomous_session_pause_event = pause_event
            self._autonomous_session_stop_event = stop_event
            self._persist_autonomous_session_status()
            self._active_thread = threading.Thread(
                target=self._run_autonomous_session_loop,
                kwargs={
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "session_id": session_id,
                    "tick_interval_ms": int(tick_interval_ms),
                    "text_hint": str(text_hint or ""),
                    "max_ticks": (int(max_ticks) if max_ticks is not None else None),
                    "pause_event": pause_event,
                    "stop_event": stop_event,
                },
                daemon=True,
                name=f"autonomous-session-{run_id}",
            )
            self._active_thread.start()
            return {"session_id": session_id, "run_id": run_id, "run_dir": str(run_dir), "status": "queued"}

    def pause_autonomous_session(self) -> dict[str, Any]:
        with self._lock:
            if not self._autonomous_session_status or not self._autonomous_session_pause_event:
                return {"ok": False, "error": "no_active_session"}
            current_status = str(self._autonomous_session_status.get("status", "") or "")
            if current_status in {"paused", "pausing"}:
                return {"ok": True, "status": current_status, "session_id": self._autonomous_session_status["session_id"]}
            if current_status not in {"queued", "running", "recovering"}:
                return {"ok": False, "error": f"session_not_pauseable:{current_status}"}
            self._autonomous_session_pause_event.set()
            self._autonomous_session_status["paused"] = True
            self._autonomous_session_status["status"] = "pausing"
            self._autonomous_session_status["updated_at_ms"] = now_ms()
            self._mark_autonomous_session_transition(self._autonomous_session_status, transition="pause_requested")
            self._persist_autonomous_session_status()
            append_jsonl(
                Path(self._autonomous_session_status["run_dir"]) / "system" / "events.jsonl",
                {"ts_ms": now_ms(), "type": "session_pause_requested", "session_id": self._autonomous_session_status["session_id"]},
            )
            return {"ok": True, "status": "pausing", "session_id": self._autonomous_session_status["session_id"]}

    def resume_autonomous_session(self) -> dict[str, Any]:
        with self._lock:
            if not self._autonomous_session_status or not self._autonomous_session_pause_event:
                return {"ok": False, "error": "no_active_session"}
            current_status = str(self._autonomous_session_status.get("status", "") or "")
            paused = bool(self._autonomous_session_status.get("paused", False))
            pause_requested = self._autonomous_session_pause_event.is_set()
            if current_status == "running" and not paused and not pause_requested:
                return {"ok": True, "status": "running", "session_id": self._autonomous_session_status["session_id"]}
            if current_status not in {"paused", "pausing", "running", "queued", "recovering"} and not pause_requested:
                return {"ok": False, "error": f"session_not_resumable:{current_status}"}
            self._autonomous_session_pause_event.clear()
            self._autonomous_session_status["paused"] = False
            self._autonomous_session_status["status"] = "running"
            self._autonomous_session_status["updated_at_ms"] = now_ms()
            self._mark_autonomous_session_transition(self._autonomous_session_status, transition="resumed")
            self._persist_autonomous_session_status()
            append_jsonl(
                Path(self._autonomous_session_status["run_dir"]) / "system" / "events.jsonl",
                {"ts_ms": now_ms(), "type": "session_resumed", "session_id": self._autonomous_session_status["session_id"]},
            )
            return {"ok": True, "status": "running", "session_id": self._autonomous_session_status["session_id"]}

    def stop_autonomous_session(self) -> dict[str, Any]:
        with self._lock:
            if not self._autonomous_session_status or not self._autonomous_session_stop_event:
                return {"ok": False, "error": "no_active_session"}
            current_status = str(self._autonomous_session_status.get("status", "") or "")
            if current_status == "stopping":
                return {"ok": True, "status": "stopping", "session_id": self._autonomous_session_status["session_id"]}
            if current_status in {"completed", "failed", "stopped"}:
                return {"ok": False, "error": f"session_already_final:{current_status}"}
            self._autonomous_session_stop_event.set()
            self._autonomous_session_status["stopping"] = True
            self._autonomous_session_status["status"] = "stopping"
            self._autonomous_session_status["updated_at_ms"] = now_ms()
            self._mark_autonomous_session_transition(self._autonomous_session_status, transition="stop_requested")
            self._persist_autonomous_session_status()
            append_jsonl(
                Path(self._autonomous_session_status["run_dir"]) / "system" / "events.jsonl",
                {"ts_ms": now_ms(), "type": "session_stop_requested", "session_id": self._autonomous_session_status["session_id"]},
            )
            return {"ok": True, "status": "stopping", "session_id": self._autonomous_session_status["session_id"]}

    def recover_autonomous_session(self, *, run_id: str | None = None, tick_interval_ms: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise AppError("已有运行中的 run 或 autonomous session，请等待其完成后再恢复。")
            target_run_id = str(run_id or "").strip() or self._find_recoverable_autonomous_session_run_id()
            if not target_run_id:
                raise AppError("没有可恢复的 autonomous session")
            run_dir = self.layout.runs_root / target_run_id
            manifest = self._read_manifest_raw(target_run_id)
            if not manifest:
                raise AppError("目标 autonomous session run 不存在")
            status_payload = self._read_autonomous_session_status_file(run_dir)
            if not status_payload:
                raise AppError("目标 autonomous session 缺少状态文件")
            checkpoint_path = self._autonomous_session_checkpoint_path(run_dir)
            if not checkpoint_path.exists():
                raise AppError("目标 autonomous session 缺少 runtime checkpoint")
            self.load_checkpoint(checkpoint_path)
            pause_event = threading.Event()
            stop_event = threading.Event()
            session_id = str(status_payload.get("session_id", "") or f"session::{target_run_id}")
            tick_done = int(status_payload.get("tick_done", 0) or 0)
            max_ticks = int(status_payload.get("max_ticks", 0) or 0)
            status_payload["active"] = True
            status_payload["paused"] = False
            status_payload["stopping"] = False
            status_payload["status"] = "recovering"
            status_payload["updated_at_ms"] = now_ms()
            if tick_interval_ms is not None:
                status_payload["tick_interval_ms"] = int(tick_interval_ms)
            status_payload = self._ensure_autonomous_session_status_defaults(status_payload)
            recover_health = dict((status_payload.get("session_health", {}) or {}))
            recover_health["health_status"] = "recovering"
            recover_health["health_reason"] = "recovering_from_runtime_checkpoint"
            recover_health["recover_hint"] = f"正在恢复，准备从本地 tick {tick_done} 继续"
            recover_health["updated_at_ms"] = now_ms()
            status_payload["session_health"] = recover_health
            self._mark_autonomous_session_transition(status_payload, transition="recovered")
            self._active_run_id = target_run_id
            self._active_session_id = session_id
            self._autonomous_session_status = status_payload
            self._autonomous_session_pause_event = pause_event
            self._autonomous_session_stop_event = stop_event
            self._persist_autonomous_session_status()
            self._active_thread = threading.Thread(
                target=self._run_autonomous_session_loop,
                kwargs={
                    "run_id": target_run_id,
                    "run_dir": run_dir,
                    "session_id": session_id,
                    "tick_interval_ms": int(status_payload.get("tick_interval_ms", 0) or 0),
                    "text_hint": str(status_payload.get("text_hint", "") or ""),
                    "max_ticks": (max_ticks or None),
                    "pause_event": pause_event,
                    "stop_event": stop_event,
                },
                daemon=True,
                name=f"autonomous-session-recover-{target_run_id}",
            )
            self._active_thread.start()
            append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": now_ms(), "type": "session_recovered", "session_id": session_id, "tick_done": tick_done})
            return {"ok": True, "session_id": session_id, "run_id": target_run_id, "status": "recovering", "tick_done": tick_done}

    def start_image_stream_run(
        self,
        *,
        frame_bytes_list: list[bytes] | None = None,
        strip_image_bytes: bytes | None = None,
        frame_count: int | None = None,
        text_prefix: str = "",
        label: str | None = None,
        tick_interval_ms: int = 0,
        reset_runtime: bool = False,
    ) -> dict[str, Any]:
        image_source = self._stream_adapter.build_image_sequence_source(
            frames=list(frame_bytes_list or []),
            strip_image_bytes=strip_image_bytes,
            frame_count=max(1, int(frame_count or 1)),
            source_type="image_stream",
        )
        if image_source.status().get("total_items", 0) == 0:
            raise AppError("image stream 需要 frame_bytes_list 或 strip_image_bytes")
        return self.start_realtime_source_run(
            source=image_source,
            text_prefix=text_prefix,
            label=label or "Phase11 连续图像流运行",
            tick_interval_ms=tick_interval_ms,
            reset_runtime=reset_runtime,
            run_kind="phase11_image_stream_run",
            notes=[
                "这是 AP 二期连续图像流适配运行。",
                "图像序列先切成逐 tick 帧，再复用现有视觉感受器与主链。",
            ],
        )

    def wait_for_idle(self, timeout_sec: float = 10.0) -> bool:
        end_time = time.time() + float(timeout_sec)
        while time.time() < end_time:
            thread = self._active_thread
            if thread is None or not thread.is_alive():
                return True
            time.sleep(0.05)
        return False

    def _persist_autonomous_session_status(self) -> None:
        status = self._ensure_autonomous_session_status_defaults(self._autonomous_session_status or {})
        run_dir_value = str(status.get("run_dir", "") or "")
        if not run_dir_value:
            return
        write_json(Path(run_dir_value) / "live" / "autonomous_session_status.json", status)

    def _update_autonomous_session_runtime_snapshot(
        self,
        *,
        runtime_tick_index: int,
        summary: dict[str, Any],
        sidecar: dict[str, Any],
        metrics: dict[str, Any],
        sandbox_result: dict[str, Any],
        teacher_review: dict[str, Any],
        screenshot_meta: dict[str, Any] | None,
        sleep_ms: int,
        autonomous_state: dict[str, int],
    ) -> None:
        with self._lock:
            if not self._autonomous_session_status:
                return
            context = dict((self._autonomous_session_status.get("session_context", {}) or {}))
            context["last_tick_id"] = str(summary.get("tick_id", "") or "")
            context["last_input_preview"] = str(summary.get("input_preview", "") or "")[:160]
            context["last_focus_preview"] = [str(item or "") for item in (summary.get("a_focus_preview", []) or []) if str(item or "")][:8]
            context["last_bn_ids"] = [
                str(item.get("memory_id", "") or "")
                for item in (sidecar.get("bn_list", []) or [])[:8]
                if str(item.get("memory_id", "") or "")
            ]
            selected_actions = list(sandbox_result.get("selected_actions", []) or [])
            context["last_selected_action_names"] = [
                str(item.get("action_name", "") or "")
                for item in selected_actions
                if str(item.get("action_name", "") or "")
            ][:8]
            context["last_selected_action_statuses"] = [
                str(item.get("status", "") or "")
                for item in selected_actions
                if str(item.get("status", "") or "")
            ][:8]
            context["last_teacher_mode"] = str(teacher_review.get("mode", "") or "")
            external_teacher_review = dict((teacher_review.get("external_teacher_review", {}) or {}))
            context["last_external_teacher_mode"] = str(
                external_teacher_review.get("mode", "")
                or external_teacher_review.get("provider", "")
                or ""
            )
            context["updated_at_ms"] = now_ms()
            self._autonomous_session_status["session_context"] = context

            health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
            health["idle_ticks"] = int(autonomous_state.get("idle_ticks", 0) or 0)
            health["capture_failures"] = int(autonomous_state.get("capture_failures", 0) or 0)
            health["action_errors"] = int(autonomous_state.get("action_errors", 0) or 0)
            health["last_logic_ms"] = round(float(metrics.get("logic_ms", 0.0) or 0.0), 4)
            health["last_sleep_ms"] = int(max(0, int(sleep_ms or 0)))
            health["last_screen_capture_ok"] = bool((screenshot_meta or {}).get("captured", False))
            health["last_tick_generated_at_ms"] = int(summary.get("generated_at_ms", 0) or 0)
            health["updated_at_ms"] = now_ms()
            self._autonomous_session_status["session_health"] = health
            self._autonomous_session_status["last_runtime_tick_index"] = int(runtime_tick_index)
            self._autonomous_session_status["updated_at_ms"] = now_ms()
            self._update_autonomous_session_progress_fields(self._autonomous_session_status)
            self._persist_autonomous_session_status()

    def _autonomous_session_checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "live" / "autonomous_runtime_checkpoint.json"

    def _persist_autonomous_session_runtime_checkpoint(self, run_dir: Path) -> dict[str, Any]:
        checkpoint_path = self._autonomous_session_checkpoint_path(run_dir)
        payload = self.export_runtime()
        write_json(checkpoint_path, payload)
        with self._lock:
            if self._autonomous_session_status:
                health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                health["last_checkpoint_at_ms"] = now_ms()
                health["last_checkpoint_tick_done"] = int(self._autonomous_session_status.get("tick_done", 0) or 0)
                health["updated_at_ms"] = now_ms()
                self._autonomous_session_status["session_health"] = health
                self._autonomous_session_status["updated_at_ms"] = now_ms()
                self._update_autonomous_session_progress_fields(self._autonomous_session_status)
                self._persist_autonomous_session_status()
        return {"ok": True, "path": str(checkpoint_path)}

    def _read_autonomous_session_status_file(self, run_dir: Path) -> dict[str, Any]:
        return read_autonomous_session_status(run_dir)

    def _find_recoverable_autonomous_session_run_id(self) -> str:
        for info in self.list_run_infos(limit=32):
            run_id = str(info.get("run_id", "") or "")
            if not run_id:
                continue
            status = self._read_autonomous_session_status_file(self.layout.runs_root / run_id)
            if not status:
                continue
            if not self._autonomous_session_checkpoint_path(self.layout.runs_root / run_id).exists():
                continue
            tick_done = int(status.get("tick_done", 0) or 0)
            max_ticks = int(status.get("max_ticks", 0) or 0)
            recoverable = bool(status.get("recoverable", True))
            if recoverable and (max_ticks <= 0 or tick_done < max_ticks):
                return run_id
        return ""

    def _normalize_bootstrap_autonomous_session_status(self, status: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(status, dict) or not status:
            return {}
        payload = self._ensure_autonomous_session_status_defaults(status)
        state = str(payload.get("status", "") or "").strip()
        active = bool(payload.get("active", False))
        if active and state in {"queued", "running", "paused", "pausing", "recovering", "stopping"}:
            payload["active"] = False
            payload["paused"] = False
            payload["stopping"] = False
            payload["recoverable"] = True
            payload["status"] = "interrupted"
            payload["last_stop_reason"] = str(payload.get("last_stop_reason", "") or "session_status_restored_without_live_thread")
            if int(payload.get("finished_at_ms", 0) or 0) <= 0:
                payload["finished_at_ms"] = now_ms()
            payload["updated_at_ms"] = now_ms()
            self._mark_autonomous_session_transition(
                payload,
                transition="interrupted",
                reason=str(payload.get("last_stop_reason", "") or "session_status_restored_without_live_thread"),
            )
            health = dict((payload.get("session_health", {}) or {}))
            health["health_status"] = "interrupted"
            health["health_reason"] = str(payload.get("last_stop_reason", "") or "session_status_restored_without_live_thread")
            health["recover_hint"] = f"可从本地 tick {int(payload.get('tick_done', 0) or 0)} 继续恢复"
            health["updated_at_ms"] = int(payload.get("updated_at_ms", now_ms()) or now_ms())
            payload["session_health"] = health
        else:
            self._update_autonomous_session_progress_fields(payload)
        return payload

    def _prepare_run_dir(self, run_id: str) -> Path:
        run_dir = make_run_dir(self.layout, run_id)
        (run_dir / "live").mkdir(parents=True, exist_ok=True)
        (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
        (run_dir / "system").mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "live" / "run_rollup.json", empty_rollup(run_id=run_id))
        write_json(run_dir / "live" / "tick_list.json", {"schema_id": "tick_list_cache/v1", "run_id": run_id, "ticks": []})
        return run_dir

    def _build_manifest(
        self,
        *,
        run_id: str,
        run_dir: Path,
        tick_planned: int,
        label: str,
        run_kind: str = "phase1_demo",
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "schema_id": "run_manifest/v1",
            "schema_version": "1.0",
            "run_id": run_id,
            "run_kind": run_kind,
            "status": "queued",
            "label": label,
            "created_at_ms": now_ms(),
            "started_at_ms": 0,
            "updated_at_ms": now_ms(),
            "finished_at_ms": 0,
            "tick_planned": int(tick_planned),
            "tick_done": 0,
            "latest_tick_index": -1,
            "paths": {
                "run_dir": str(run_dir),
                "live_dir": str(run_dir / "live"),
                "chunks_dir": str(run_dir / "chunks"),
                "system_dir": str(run_dir / "system"),
            },
            "config_snapshot": {
                "host": self.config.host,
                "port": self.config.port,
                "live_ring_limit": self.config.live_ring_limit,
                "run_chunk_size": self.config.run_chunk_size,
                "text_sensor_budget": self.config.text_sensor_budget,
                "r_state_head_limit": self.config.r_state_head_limit,
                "vision_patch_budget": self.config.vision_patch_budget,
                "vision_raw_state_budget": self.config.vision_raw_state_budget,
                "vision_focus_patch_budget": self.config.vision_focus_patch_budget,
                "vision_dynamic_track_window": self.config.vision_dynamic_track_window,
                "vision_dynamic_candidate_limit_background": self.config.vision_dynamic_candidate_limit_background,
                "vision_dynamic_candidate_limit_focus": self.config.vision_dynamic_candidate_limit_focus,
                "vision_dynamic_track_limit": self.config.vision_dynamic_track_limit,
                "vision_dynamic_summary_limit": self.config.vision_dynamic_summary_limit,
                "vision_dynamic_match_threshold": self.config.vision_dynamic_match_threshold,
                "vision_dynamic_track_forget_ticks": self.config.vision_dynamic_track_forget_ticks,
                "hearing_window_budget": self.config.hearing_window_budget,
                "executor_enabled": self.config.executor_enabled,
                "executor_dry_run": self.config.executor_dry_run,
                "autonomous_capture_required": self.config.autonomous_capture_required,
                "autonomous_auto_feedback_enabled": self.config.autonomous_auto_feedback_enabled,
                "autonomous_teacher_enabled": self.config.autonomous_teacher_enabled,
                "autonomous_teacher_mode": self.config.autonomous_teacher_mode,
                "autonomous_llm_gate_enabled": self.config.autonomous_llm_gate_enabled,
                "autonomous_llm_gate_mode": self.config.autonomous_llm_gate_mode,
            },
            "notes": notes
            or [
                "这是 AP 二期 Phase 0 / Phase 1 的最小演示运行。",
                "当前尚未接入真实 HDB-V2 与完整多模态闭环，这里用于验证 run/log/live/replay 底座。",
            ],
        }
        validate_or_raise(manifest, load_schema("run_manifest.schema.json"), label="run_manifest")
        return manifest

    def _write_manifest(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        clean_manifest = dict(manifest or {})
        clean_manifest.pop("autonomous_session_status_summary", None)
        validate_or_raise(clean_manifest, load_schema("run_manifest.schema.json"), label="run_manifest")
        write_json(run_dir / "manifest.json", clean_manifest)

    def _run_demo_loop(self, *, run_id: str, run_dir: Path, tick_count: int, tick_interval_ms: int) -> None:
        started_at_ms = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = "running"
        manifest["started_at_ms"] = started_at_ms
        manifest["updated_at_ms"] = started_at_ms
        self._write_manifest(run_dir, manifest)
        append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": started_at_ms, "type": "run_started", "run_id": run_id})
        try:
            for tick_index in range(int(tick_count)):
                tick_started = time.perf_counter()
                generated_at_ms = now_ms()
                tick_id = f"{run_id}_tick_{tick_index:06d}"
                summary = {
                    "schema_id": "tick_summary/v1",
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "tick_index": tick_index,
                    "tick_id": tick_id,
                    "status": "ok",
                    "generated_at_ms": generated_at_ms,
                    "input_preview": f"phase1_input_{tick_index}",
                    "state_top": [
                        {"sa_label": "phase1_anchor", "energy": round(1.0 - min(0.7, tick_index * 0.03), 3)},
                        {"sa_label": f"demo_tick_{tick_index}", "energy": round(0.5 + tick_index * 0.05, 3)},
                    ],
                    "r_state_preview": ["状态池现状已被固定预算采样", f"当前 tick={tick_index}"],
                    "a_focus_preview": ["显意识短片段", f"demo_focus_{tick_index}"],
                    "notes": ["该 tick 仅用于验证运行底座，不代表真实认知逻辑。", "后续 Phase 2 会将这里替换为文本输入最小闭环。"],
                }
                validate_or_raise(summary, load_schema("tick_summary.schema.json"), label="tick_summary")
                append_jsonl(chunk_file(run_dir, kind="summary", tick_index=tick_index, chunk_size=self.config.run_chunk_size), summary)

                logic_ms = round((time.perf_counter() - tick_started) * 1000 + 1.0, 3)
                metrics = {
                    "schema_id": "tick_metrics/v1",
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "tick_index": tick_index,
                    "runtime_tick_index": tick_index,
                    "logic_ms": logic_ms,
                    "live_payload_bytes": len(json.dumps({"tick_id": tick_id}, ensure_ascii=False).encode("utf-8")),
                    "state_top_count": len(summary["state_top"]),
                    "r_state_count": len(summary["r_state_preview"]),
                    "a_focus_count": len(summary["a_focus_preview"]),
                    "state_pool_size": len(summary["state_top"]),
                    "state_pool_anchor_count": 0,
                    "state_pool_residual_count": 0,
                    "bn_count": 0,
                    "c_star_count": 0,
                    "text_budget_used": 0,
                    "vision_budget_used": 0,
                    "audio_budget_used": 0,
                }
                append_jsonl(chunk_file(run_dir, kind="metrics", tick_index=tick_index, chunk_size=self.config.run_chunk_size), metrics)
                self._update_run_caches(run_dir=run_dir, summary=summary, metrics=metrics)
                self._advance_manifest(run_id=run_id, run_dir=run_dir, tick_index=tick_index)
                self._publish_live(run_id=run_id, run_dir=run_dir, summary=summary, metrics=metrics, status="running")
                time.sleep(max(0.001, tick_interval_ms / 1000.0))
            self._complete_run(run_id=run_id, run_dir=run_dir, final_tick_index=max(0, tick_count - 1))
        except Exception as exc:
            self._fail_run(run_id=run_id, run_dir=run_dir, error=str(exc))
            raise
        finally:
            with self._lock:
                self._active_run_id = ""

    def _run_multimodal_loop(
        self,
        *,
        run_id: str,
        run_dir: Path,
        items: list[dict[str, Any]],
        tick_interval_ms: int,
        base_tick_index: int,
    ) -> None:
        started_at_ms = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = "running"
        manifest["started_at_ms"] = started_at_ms
        manifest["updated_at_ms"] = started_at_ms
        self._write_manifest(run_dir, manifest)
        append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": started_at_ms, "type": "run_started", "run_id": run_id})
        autonomous_state = {
            "capture_failures": 0,
            "action_errors": 0,
            "idle_ticks": 0,
        }
        try:
            for local_tick_index, item in enumerate(items):
                runtime_tick_index = int(base_tick_index + local_tick_index)
                tick_result = self._execute_runtime_item(
                    run_id=run_id,
                    run_dir=run_dir,
                    local_tick_index=local_tick_index,
                    runtime_tick_index=runtime_tick_index,
                    item=item,
                    autonomous_state=autonomous_state,
                    tick_interval_ms=tick_interval_ms,
                )
                sleep_ms = int(tick_result.get("sleep_ms", 0) or 0)
                time.sleep(max(0.0, sleep_ms / 1000.0))

            self._complete_run(run_id=run_id, run_dir=run_dir, final_tick_index=max(0, len(items) - 1))
        except GracefulRunStop as exc:
            self._complete_run(
                run_id=run_id,
                run_dir=run_dir,
                final_tick_index=max(0, min(len(items) - 1, int(self._read_manifest_raw(run_id).get("latest_tick_index", -1) or -1))),
                completion_status="stopped",
                completion_note=exc.reason,
                completion_details=exc.details,
            )
        except Exception as exc:
            self._fail_run(run_id=run_id, run_dir=run_dir, error=str(exc))
            raise
        finally:
            with self._lock:
                self._active_run_id = ""

    def _run_realtime_source_loop(
        self,
        *,
        run_id: str,
        run_dir: Path,
        source: BaseRealtimeSourceV1,
        text_prefix: str,
        tick_interval_ms: int,
        base_tick_index: int,
    ) -> None:
        started_at_ms = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = "running"
        manifest["started_at_ms"] = started_at_ms
        manifest["updated_at_ms"] = started_at_ms
        manifest["source_meta"] = source.status()
        self._write_manifest(run_dir, manifest)
        append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": started_at_ms, "type": "run_started", "run_id": run_id})
        autonomous_state = {"capture_failures": 0, "action_errors": 0, "idle_ticks": 0}
        # Realtime source runs are independent runs and must always maintain
        # their own local tick numbering starting from 0.
        local_tick_index = 0
        try:
            while True:
                item = source.next_item()
                if item is None:
                    break
                merged_item = {"text": str(text_prefix or ""), "source_type": str(item.get("source_type", "realtime_stream") or "realtime_stream")}
                merged_item.update(dict(item))
                append_jsonl(
                    run_dir / "inputs" / "inputs.jsonl",
                    {
                        "schema_id": "multimodal_input_envelope/v1",
                        "schema_version": "1.0",
                        "tick_index": local_tick_index,
                        **self._serialize_input_item(merged_item),
                    },
                )
                runtime_tick_index = int(base_tick_index + local_tick_index)
                tick_result = self._execute_runtime_item(
                    run_id=run_id,
                    run_dir=run_dir,
                    local_tick_index=local_tick_index,
                    runtime_tick_index=runtime_tick_index,
                    item=merged_item,
                    autonomous_state=autonomous_state,
                    tick_interval_ms=tick_interval_ms,
                )
                local_tick_index += 1
                time.sleep(max(0.0, float(tick_result.get("sleep_ms", 0) or 0) / 1000.0))

            final_tick_index = max(0, local_tick_index - 1)
            if local_tick_index == 0:
                raise AppError("realtime source 未产出任何有效 item")
            self._complete_run(run_id=run_id, run_dir=run_dir, final_tick_index=final_tick_index)
        except GracefulRunStop as exc:
            self._complete_run(
                run_id=run_id,
                run_dir=run_dir,
                final_tick_index=max(0, min(max(0, local_tick_index - 1), int(self._read_manifest_raw(run_id).get("latest_tick_index", -1) or -1))),
                completion_status="stopped",
                completion_note=exc.reason,
                completion_details=exc.details,
            )
        except Exception as exc:
            self._fail_run(run_id=run_id, run_dir=run_dir, error=str(exc))
            raise
        finally:
            try:
                source.close()
            except Exception:
                pass
            with self._lock:
                self._active_run_id = ""
                self._active_stream_source = None

    def _run_autonomous_session_loop(
        self,
        *,
        run_id: str,
        run_dir: Path,
        session_id: str,
        tick_interval_ms: int,
        text_hint: str,
        max_ticks: int | None,
        pause_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        started_at_ms = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = "running"
        manifest["started_at_ms"] = started_at_ms
        manifest["updated_at_ms"] = started_at_ms
        self._write_manifest(run_dir, manifest)
        append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": started_at_ms, "type": "run_started", "run_id": run_id})
        autonomous_state = {"capture_failures": 0, "action_errors": 0, "idle_ticks": 0}
        # A recovered session must continue from the persisted local tick
        # instead of rewriting tick_000000... from scratch.
        local_tick_index = int((self._autonomous_session_status or {}).get("tick_done", 0) or 0)
        try:
            with self._lock:
                if self._autonomous_session_status:
                    self._autonomous_session_status["status"] = "running"
                    self._autonomous_session_status["started_at_ms"] = started_at_ms
                    self._autonomous_session_status["updated_at_ms"] = started_at_ms
                    health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                    if str(self._autonomous_session_status.get("status", "") or "") == "recovering":
                        health["health_status"] = "recovering"
                        health["health_reason"] = "recovering_from_runtime_checkpoint"
                    else:
                        health["health_status"] = "healthy"
                        health["health_reason"] = "session_running"
                    health["updated_at_ms"] = started_at_ms
                    self._autonomous_session_status["session_health"] = health
                    self._update_autonomous_session_progress_fields(self._autonomous_session_status)
                    self._persist_autonomous_session_status()
            while True:
                if stop_event.is_set():
                    raise GracefulRunStop("自主 session 收到停止请求", details={"stop_requested": True})
                if max_ticks is not None and local_tick_index >= int(max_ticks):
                    break
                while pause_event.is_set():
                    with self._lock:
                        if self._autonomous_session_status:
                            already_paused = (
                                str(self._autonomous_session_status.get("status", "") or "") == "paused"
                                and bool(self._autonomous_session_status.get("paused", False))
                            )
                            if not already_paused:
                                self._autonomous_session_status["status"] = "paused"
                                self._autonomous_session_status["paused"] = True
                                self._autonomous_session_status["updated_at_ms"] = now_ms()
                                health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                                health["health_status"] = "paused"
                                health["health_reason"] = "session_paused_by_operator"
                                health["updated_at_ms"] = now_ms()
                                self._autonomous_session_status["session_health"] = health
                                self._mark_autonomous_session_transition(self._autonomous_session_status, transition="paused")
                                self._persist_autonomous_session_status()
                    if stop_event.is_set():
                        raise GracefulRunStop("自主 session 收到停止请求", details={"stop_requested": True})
                    time.sleep(0.05)
                runtime_tick_index = int((self._autonomous_session_status.get("base_tick_index", 0) if self._autonomous_session_status else 0) + local_tick_index)
                tick_meta = dict((self._autonomous_session_status or {}).get("autonomous_tick_meta", {}) or {})
                tick_meta["planned_tick_index"] = local_tick_index
                item = {
                    "text": str(text_hint or ""),
                    "source_type": "autonomous_loop",
                    "capture_screen": True,
                    "autonomous_tick_meta": tick_meta,
                }
                tick_result = self._execute_runtime_item(
                    run_id=run_id,
                    run_dir=run_dir,
                    local_tick_index=local_tick_index,
                    runtime_tick_index=runtime_tick_index,
                    item=item,
                    autonomous_state=autonomous_state,
                    tick_interval_ms=tick_interval_ms,
                )
                self._update_autonomous_session_runtime_snapshot(
                    runtime_tick_index=runtime_tick_index,
                    summary=dict(tick_result.get("summary", {}) or {}),
                    sidecar=dict(tick_result.get("sidecar", {}) or {}),
                    metrics=dict(tick_result.get("metrics", {}) or {}),
                    sandbox_result=dict(((tick_result.get("sidecar", {}) or {}).get("sandbox_result", {}) or {})),
                    teacher_review=dict(((tick_result.get("sidecar", {}) or {}).get("teacher_review", {}) or {})),
                    screenshot_meta=dict(item.get("_screenshot_meta", {}) or {}),
                    sleep_ms=int(tick_result.get("sleep_ms", 0) or 0),
                    autonomous_state=dict(autonomous_state),
                )
                with self._lock:
                    if self._autonomous_session_status:
                        self._autonomous_session_status["status"] = "running"
                        self._autonomous_session_status["paused"] = False
                        self._autonomous_session_status["tick_done"] = local_tick_index + 1
                        self._autonomous_session_status["last_tick_index"] = local_tick_index
                        self._autonomous_session_status["last_runtime_tick_index"] = runtime_tick_index
                        self._autonomous_session_status["updated_at_ms"] = now_ms()
                        self._update_autonomous_session_progress_fields(self._autonomous_session_status)
                        self._persist_autonomous_session_status()
                self._persist_autonomous_session_runtime_checkpoint(run_dir)
                local_tick_index += 1
                time.sleep(max(0.0, float(tick_result.get("sleep_ms", 0) or 0) / 1000.0))
            self._complete_run(run_id=run_id, run_dir=run_dir, final_tick_index=max(0, local_tick_index - 1))
            with self._lock:
                if self._autonomous_session_status:
                    self._autonomous_session_status["active"] = False
                    self._autonomous_session_status["status"] = "completed"
                    self._autonomous_session_status["recoverable"] = False
                    self._autonomous_session_status["finished_at_ms"] = now_ms()
                    self._autonomous_session_status["updated_at_ms"] = now_ms()
                    health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                    health["health_status"] = "completed"
                    health["health_reason"] = "target_completed"
                    health["updated_at_ms"] = now_ms()
                    self._autonomous_session_status["session_health"] = health
                    self._mark_autonomous_session_transition(self._autonomous_session_status, transition="completed")
                    self._persist_autonomous_session_status()
        except GracefulRunStop as exc:
            final_tick_index = max(0, min(max(0, local_tick_index - 1), int(self._read_manifest_raw(run_id).get("latest_tick_index", -1) or -1)))
            self._complete_run(
                run_id=run_id,
                run_dir=run_dir,
                final_tick_index=final_tick_index,
                completion_status="stopped",
                completion_note=exc.reason,
                completion_details=exc.details,
            )
            with self._lock:
                if self._autonomous_session_status:
                    self._autonomous_session_status["active"] = False
                    self._autonomous_session_status["status"] = "stopped"
                    self._autonomous_session_status["recoverable"] = True
                    self._autonomous_session_status["finished_at_ms"] = now_ms()
                    self._autonomous_session_status["updated_at_ms"] = now_ms()
                    self._autonomous_session_status["last_stop_reason"] = exc.reason
                    health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                    health["health_status"] = "stopped"
                    health["health_reason"] = str(exc.reason or "session_stopped")
                    health["updated_at_ms"] = now_ms()
                    self._autonomous_session_status["session_health"] = health
                    self._mark_autonomous_session_transition(self._autonomous_session_status, transition="stopped", reason=exc.reason)
                    self._persist_autonomous_session_status()
        except Exception as exc:
            self._fail_run(run_id=run_id, run_dir=run_dir, error=str(exc))
            with self._lock:
                if self._autonomous_session_status:
                    self._autonomous_session_status["active"] = False
                    self._autonomous_session_status["status"] = "failed"
                    self._autonomous_session_status["recoverable"] = True
                    self._autonomous_session_status["finished_at_ms"] = now_ms()
                    self._autonomous_session_status["updated_at_ms"] = now_ms()
                    self._autonomous_session_status["last_stop_reason"] = str(exc)
                    health = dict((self._autonomous_session_status.get("session_health", {}) or {}))
                    health["health_status"] = "failed"
                    health["health_reason"] = str(exc)
                    health["updated_at_ms"] = now_ms()
                    self._autonomous_session_status["session_health"] = health
                    self._mark_autonomous_session_transition(self._autonomous_session_status, transition="failed", reason=str(exc))
                    self._persist_autonomous_session_status()
            raise
        finally:
            with self._lock:
                self._active_run_id = ""
                self._active_session_id = ""
                self._autonomous_session_pause_event = None
                self._autonomous_session_stop_event = None

    def _execute_runtime_item(
        self,
        *,
        run_id: str,
        run_dir: Path,
        local_tick_index: int,
        runtime_tick_index: int,
        item: dict[str, Any],
        autonomous_state: dict[str, int],
        tick_interval_ms: int,
    ) -> dict[str, Any]:
        tick_started = time.perf_counter()
        tick_id = f"{run_id}_tick_{local_tick_index:06d}"
        text = str(item.get("text", "") or "")
        source_type = str(item.get("source_type", "multimodal_input") or "multimodal_input")
        text_packet = self._runtime.text_sensor.ingest(text, tick_index=runtime_tick_index, source_type=source_type)
        image_packet = None
        audio_packet = None
        if item.get("image_bytes") is not None:
            image_packet = self._runtime.vision_sensor.ingest_image_bytes(bytes(item.get("image_bytes") or b""), tick_index=runtime_tick_index, source_type="image_input")
        elif bool(item.get("capture_screen", False)):
            screenshot = self._agent_sandbox.capture_screenshot_packet()
            if self._autonomous_mode_enabled(item):
                self._check_autonomous_capture(screenshot=screenshot, item=item, state=autonomous_state)
            if bool(screenshot.get("captured", False)) and screenshot.get("image_bytes") is not None:
                image_packet = self._runtime.vision_sensor.ingest_image_bytes(bytes(screenshot.get("image_bytes") or b""), tick_index=runtime_tick_index, source_type="screen_capture")
            item["_screenshot_meta"] = screenshot
        if item.get("audio_bytes") is not None:
            audio_packet = self._runtime.hearing_sensor.ingest_wav_bytes(bytes(item.get("audio_bytes") or b""), tick_index=runtime_tick_index, source_type="audio_input")

        runtime_tick = self._runtime.process_multimodal_tick(
            tick_index=runtime_tick_index,
            text_packet=text_packet,
            image_packet=image_packet,
            audio_packet=audio_packet,
            source_type=source_type,
        )
        teacher_review = self._runtime.teacher_layer.review_actions(
            tick_index=runtime_tick_index,
            action_drives=runtime_tick.get("rules_result", {}).get("planned_selected_actions_preview", []) or runtime_tick.get("rules_result", {}).get("action_drives", []) or [],
            runtime_tick=runtime_tick,
            autonomous_state=autonomous_state if self._autonomous_mode_enabled(item) else None,
            teacher_mode_override=str((item.get("autonomous_tick_meta", {}) or {}).get("teacher_mode", "") or "") or None,
            llm_gate_mode_override=str((item.get("autonomous_tick_meta", {}) or {}).get("llm_gate_mode", "") or "") or None,
            external_teacher_enabled_override=(
                (item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_enabled")
                if "external_teacher_enabled" in (item.get("autonomous_tick_meta", {}) or {})
                else None
            ),
            external_teacher_mode_override=str((item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_mode", "") or "") or None,
            external_teacher_stub_response_path_override=str(
                (item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_stub_response_path", "") or ""
            )
            or None,
            external_teacher_fail_open_override=(
                (item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_fail_open")
                if "external_teacher_fail_open" in (item.get("autonomous_tick_meta", {}) or {})
                else None
            ),
            external_teacher_max_retries_override=(
                int((item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_max_retries", 0) or 0)
                if "external_teacher_max_retries" in (item.get("autonomous_tick_meta", {}) or {})
                else None
            ),
            external_teacher_retry_backoff_ms_override=(
                int((item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_retry_backoff_ms", 0) or 0)
                if "external_teacher_retry_backoff_ms" in (item.get("autonomous_tick_meta", {}) or {})
                else None
            ),
            external_teacher_http_endpoint_override=str(
                (item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_http_endpoint", "") or ""
            )
            or None,
            external_teacher_http_headers_override=(
                dict((item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_http_headers", {}) or {})
                if isinstance((item.get("autonomous_tick_meta", {}) or {}).get("external_teacher_http_headers"), dict)
                else None
            ),
        )
        runtime_tick["teacher_review"] = teacher_review
        executable_actions = teacher_review.get("scored_action_drives", []) or runtime_tick.get("rules_result", {}).get("planned_selected_actions_preview", []) or []
        sandbox_result = self._agent_sandbox.evaluate_action_drives(
            tick_index=runtime_tick_index,
            action_drives=executable_actions,
        )
        runtime_action_effects = self._runtime.apply_selected_actions(
            sandbox_result.get("selected_actions", []) or [],
            runtime_tick=runtime_tick,
        )
        external_feedback = dict(item.get("external_feedback", {}) or {})
        operator_feedback = dict(external_feedback)
        if self._autonomous_mode_enabled(item):
            auto_feedback = self._build_autonomous_auto_feedback(
                item=item,
                sandbox_result=sandbox_result,
                runtime_action_effects=runtime_action_effects,
                screenshot_meta=dict(item.get("_screenshot_meta", {}) or {}),
                state=autonomous_state,
            )
            if auto_feedback:
                external_feedback = self._merge_feedback(external_feedback, auto_feedback)
        teacher_feedback = self._runtime.teacher_layer.build_teacher_feedback(
            tick_index=runtime_tick_index,
            runtime_tick=runtime_tick,
            teacher_review=teacher_review,
            selected_actions=sandbox_result.get("selected_actions", []) or [],
            sandbox_result=sandbox_result,
            runtime_action_effects=runtime_action_effects,
        )
        intrinsic_feedback = self._runtime.build_intrinsic_feedback(
            emotion_channels=runtime_tick.get("rules_result", {}).get("emotion_channels", {}) or {},
            balance_metrics={
                "alignment_score": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_alignment_score", 0.0) or 0.0)),
                "grasp_score": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_grasp_score", 0.0) or 0.0)),
                "overprediction_ratio": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_overprediction_ratio", 0.0) or 0.0)),
                "underprediction_ratio": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_underprediction_ratio", 0.0) or 0.0)),
                "committed_alignment_score": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_committed_alignment_score", 0.0) or 0.0)),
                "committed_grasp_score": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_committed_grasp_score", 0.0) or 0.0)),
                "committed_overprediction_ratio": float((((runtime_tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {}).get("state.prediction_committed_overprediction_ratio", 0.0) or 0.0)),
            },
        )
        merged_feedback = self._runtime.merge_feedback_channels(
            external_feedback=external_feedback,
            teacher_feedback=teacher_feedback,
            intrinsic_feedback=intrinsic_feedback,
        )
        action_feedback = self._runtime.apply_action_feedback(
            tick_index=runtime_tick_index,
            selected_actions=sandbox_result.get("selected_actions", []) or [],
            emotion_channels=runtime_tick.get("rules_result", {}).get("emotion_channels", {}) or {},
            runtime_action_effects=runtime_action_effects,
            external_feedback=merged_feedback,
        )
        teacher_provenance = {
            "selected_action_ids": [
                str(action.get("action_id", "") or "")
                for action in (sandbox_result.get("selected_actions", []) or [])
                if str(action.get("action_id", "") or "")
            ],
            "selected_action_names": [
                str(action.get("action_name", "") or "")
                for action in (sandbox_result.get("selected_actions", []) or [])
                if str(action.get("action_name", "") or "")
            ],
            "bn_ids": [
                str(item.get("memory_id", "") or "")
                for item in (runtime_tick.get("bn_list", []) or [])[:6]
                if str(item.get("memory_id", "") or "")
            ],
            "focus_memory_id": str((runtime_tick.get("focus_memory", {}) or {}).get("memory_id", "") or ""),
            "exact_memory_id": str((runtime_tick.get("exact_memory", {}) or {}).get("memory_id", "") or ""),
            "rule_ids": [
                str(item.get("rule_id", "") or "")
                for item in ((runtime_tick.get("rules_result", {}) or {}).get("rules_fired", []) or [])
                if str(item.get("rule_id", "") or "")
            ],
            "teacher_warning_codes": [
                str(item.get("code", "") or "")
                for item in (teacher_review.get("warnings", []) or [])
                if str(item.get("code", "") or "")
            ],
        }
        feedback_signal_result = self._runtime.inject_feedback_signals(
            tick_index=runtime_tick_index,
            feedback=merged_feedback,
            provenance=teacher_provenance,
            source_type="autonomous_feedback" if self._autonomous_mode_enabled(item) else "external_feedback",
            channel="autonomous_feedback" if self._autonomous_mode_enabled(item) else "external_feedback",
            meta_extra={
                "teacher_review": dict((teacher_feedback or {}).get("teacher_review", {}) or {}),
                "external_teacher_review": dict((teacher_feedback or {}).get("external_teacher_review", {}) or {}),
                "merged_feedback": dict(merged_feedback),
                "operator_feedback": dict(operator_feedback),
                "intrinsic_feedback": dict(intrinsic_feedback),
            },
        )
        teacher_feedback_result = {
            "reward": round(float(teacher_feedback.get("reward", 0.0) or 0.0), 4),
            "punishment": round(float(teacher_feedback.get("punishment", 0.0) or 0.0), 4),
            "notes": list(teacher_feedback.get("notes", []) or []),
            "sources": copy.deepcopy((merged_feedback.get("sources", {}) or {}).get("teacher", {}) or {}),
            "injected_items": [],
            "pool_result": {},
            "teacher_review": dict(teacher_feedback.get("teacher_review", {}) or {}),
            "external_teacher_review": dict(teacher_feedback.get("external_teacher_review", {}) or {}),
            "teacher_provenance": teacher_provenance,
        }
        intrinsic_feedback_result = {
            "reward": round(float(intrinsic_feedback.get("reward", 0.0) or 0.0), 4),
            "punishment": round(float(intrinsic_feedback.get("punishment", 0.0) or 0.0), 4),
            "notes": list(intrinsic_feedback.get("notes", []) or []),
            "enabled": bool(intrinsic_feedback.get("enabled", False)),
            "current_emotion": dict(intrinsic_feedback.get("current_emotion", {}) or {}),
            "previous_emotion": dict(intrinsic_feedback.get("previous_emotion", {}) or {}),
            "delta_emotion": dict(intrinsic_feedback.get("delta_emotion", {}) or {}),
            "components": dict(intrinsic_feedback.get("components", {}) or {}),
        }
        if self._autonomous_mode_enabled(item):
            self._update_autonomous_counters(
                state=autonomous_state,
                sandbox_result=sandbox_result,
                runtime_action_effects=runtime_action_effects,
            )
        summary, sidecar, metrics = self._build_runtime_tick_artifacts(
            run_id=run_id,
            tick_id=tick_id,
            tick_index=local_tick_index,
            runtime_tick_index=runtime_tick_index,
            input_preview=text[:120] if text else self._build_multimodal_preview(item),
            runtime_tick=runtime_tick,
            sandbox_result=sandbox_result,
            runtime_action_effects=runtime_action_effects,
            action_feedback=action_feedback,
            teacher_review=teacher_review,
            teacher_feedback=teacher_feedback_result,
            input_item=item,
        )
        if self._autonomous_mode_enabled(item):
            summary["autonomous_summary"] = {
                "capture_failures": int(autonomous_state.get("capture_failures", 0) or 0),
                "action_errors": int(autonomous_state.get("action_errors", 0) or 0),
                "idle_ticks": int(autonomous_state.get("idle_ticks", 0) or 0),
                "auto_feedback_applied": dict(external_feedback),
                "merged_feedback": dict(merged_feedback),
                "intrinsic_feedback": dict(intrinsic_feedback_result),
                "feedback_signal_result": dict(feedback_signal_result),
                "teacher_feedback": dict(teacher_feedback_result),
            }
            sidecar["autonomous_sidecar"] = {
                "state": dict(autonomous_state),
                "feedback_used": dict(external_feedback),
                "operator_feedback": dict(operator_feedback),
                "merged_feedback": dict(merged_feedback),
                "intrinsic_feedback": dict(intrinsic_feedback_result),
                "feedback_signal_result": dict(feedback_signal_result),
                "teacher_review": dict(teacher_review),
                "teacher_feedback": dict(teacher_feedback_result),
                "tick_meta": dict(item.get("autonomous_tick_meta", {}) or {}),
                "screen_capture_meta": dict((item.get("_screenshot_meta", {}) or {}).get("meta", {}) or {}),
            }

        append_jsonl(chunk_file(run_dir, kind="summary", tick_index=local_tick_index, chunk_size=self.config.run_chunk_size), summary)
        storage_sidecar, externalized_rows = self._build_storage_sidecar(sidecar, run_dir=run_dir, run_id=run_id, tick_index=local_tick_index)
        append_jsonl(
            chunk_file(run_dir, kind="sidecar", tick_index=local_tick_index, chunk_size=self.config.run_chunk_size),
            storage_sidecar,
        )
        append_jsonl(chunk_file(run_dir, kind="sensor", tick_index=local_tick_index, chunk_size=self.config.run_chunk_size), text_packet)
        for kind, payload in externalized_rows.items():
            append_jsonl(chunk_file(run_dir, kind=kind, tick_index=local_tick_index, chunk_size=self.config.run_chunk_size), payload)
        metrics["logic_ms"] = round((time.perf_counter() - tick_started) * 1000 + 1.0, 3)
        self._runtime.set_last_logic_ms(float(metrics["logic_ms"] or 0.0))
        append_jsonl(chunk_file(run_dir, kind="metrics", tick_index=local_tick_index, chunk_size=self.config.run_chunk_size), metrics)
        self._update_run_caches(run_dir=run_dir, summary=summary, metrics=metrics)
        self._advance_manifest(run_id=run_id, run_dir=run_dir, tick_index=local_tick_index)
        self._publish_live(run_id=run_id, run_dir=run_dir, summary=summary, metrics=metrics, status="running")

        sleep_ms = max(0, int(tick_interval_ms))
        if self._autonomous_mode_enabled(item):
            tick_meta = dict(item.get("autonomous_tick_meta", {}) or {})
            if int(autonomous_state.get("idle_ticks", 0) or 0) > 0:
                sleep_ms = max(sleep_ms, int(tick_meta.get("idle_backoff_ms", self.config.autonomous_idle_backoff_ms) or self.config.autonomous_idle_backoff_ms))
            self._check_autonomous_stop_conditions(item=item, state=autonomous_state)
        return {
            "summary": summary,
            "sidecar": sidecar,
            "metrics": metrics,
            "sleep_ms": sleep_ms,
        }

    def _autonomous_mode_enabled(self, item: dict[str, Any]) -> bool:
        return str(item.get("source_type", "") or "") == "autonomous_loop"

    def _merge_feedback(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        for key in ("reward", "punishment"):
            merged[key] = float(merged.get(key, 0.0) or 0.0) + float(extra.get(key, 0.0) or 0.0)
        if extra:
            notes = list(merged.get("notes", []) or [])
            notes.extend(list(extra.get("notes", []) or []))
            if notes:
                merged["notes"] = notes
        return merged

    def _feedback_source_metric(self, breakdown: dict[str, Any] | None, source_name: str, metric: str) -> float:
        payload = dict(breakdown or {})
        sources = dict(payload.get("sources", {}) or {})
        source = dict(sources.get(str(source_name or ""), {}) or {})
        return float(source.get(str(metric or ""), 0.0) or 0.0)

    def _check_autonomous_capture(self, *, screenshot: dict[str, Any], item: dict[str, Any], state: dict[str, int]) -> None:
        required = bool(self.config.autonomous_capture_required)
        if not bool(screenshot.get("captured", False)):
            state["capture_failures"] = int(state.get("capture_failures", 0) or 0) + 1
            if required and int(state["capture_failures"]) >= int((item.get("autonomous_tick_meta", {}) or {}).get("stop_on_capture_failures", self.config.autonomous_stop_on_consecutive_capture_failures)):
                raise GracefulRunStop(
                    "连续截图失败，已主动停止自主循环",
                    details={"capture_failures": int(state["capture_failures"])},
                )
        else:
            state["capture_failures"] = 0

    def _update_autonomous_counters(
        self,
        *,
        state: dict[str, int],
        sandbox_result: dict[str, Any],
        runtime_action_effects: dict[str, Any],
    ) -> None:
        selected = list(sandbox_result.get("selected_actions", []) or [])
        errors = [item for item in selected if str(item.get("status", "") or "") == "error"]
        if errors:
            state["action_errors"] = int(state.get("action_errors", 0) or 0) + len(errors)
        else:
            state["action_errors"] = 0
        meaningful_move = bool(runtime_action_effects.get("moved", False))
        if not selected or (not meaningful_move and all(str(item.get("effect", "") or "") in {"wait", "noop"} for item in selected)):
            state["idle_ticks"] = int(state.get("idle_ticks", 0) or 0) + 1
        else:
            state["idle_ticks"] = 0

    def _check_autonomous_stop_conditions(self, *, item: dict[str, Any], state: dict[str, int]) -> None:
        tick_meta = dict(item.get("autonomous_tick_meta", {}) or {})
        max_action_errors = int(tick_meta.get("stop_on_action_errors", self.config.autonomous_stop_on_consecutive_action_errors) or self.config.autonomous_stop_on_consecutive_action_errors)
        max_idle = int(tick_meta.get("stop_on_idle_ticks", self.config.autonomous_stop_on_consecutive_idle_ticks) or self.config.autonomous_stop_on_consecutive_idle_ticks)
        if int(state.get("action_errors", 0) or 0) >= max_action_errors:
            raise GracefulRunStop(
                "连续动作执行错误过多，已主动停止自主循环",
                details={"action_errors": int(state.get("action_errors", 0) or 0)},
            )
        if int(state.get("idle_ticks", 0) or 0) >= max_idle:
            raise GracefulRunStop(
                "连续空转过多，已主动停止自主循环",
                details={"idle_ticks": int(state.get("idle_ticks", 0) or 0)},
            )

    def _build_autonomous_auto_feedback(
        self,
        *,
        item: dict[str, Any],
        sandbox_result: dict[str, Any],
        runtime_action_effects: dict[str, Any],
        screenshot_meta: dict[str, Any],
        state: dict[str, int],
    ) -> dict[str, Any]:
        tick_meta = dict(item.get("autonomous_tick_meta", {}) or {})
        if not bool(tick_meta.get("auto_feedback_enabled", self.config.autonomous_auto_feedback_enabled)):
            return {}
        reward = 0.0
        punishment = 0.0
        notes: list[str] = []
        selected = list(sandbox_result.get("selected_actions", []) or [])
        if bool(screenshot_meta.get("captured", False)):
            reward += 0.03
            notes.append("screen_capture_ok")
        else:
            punishment += 0.08
            notes.append("screen_capture_failed")
        if bool(runtime_action_effects.get("moved", False)):
            reward += 0.06
            notes.append("gaze_or_cursor_moved")
        if selected:
            reward += 0.02
            notes.append("action_selected")
        else:
            punishment += 0.02
            notes.append("no_action_selected")
        if int(state.get("idle_ticks", 0) or 0) > 0:
            punishment += min(0.08, int(state.get("idle_ticks", 0) or 0) * 0.01)
            notes.append("idle_tick_penalty")
        if int(state.get("action_errors", 0) or 0) > 0:
            punishment += min(0.18, int(state.get("action_errors", 0) or 0) * 0.04)
            notes.append("action_error_penalty")
        payload: dict[str, Any] = {}
        if reward > 0:
            payload["reward"] = round(reward, 4)
        if punishment > 0:
            payload["punishment"] = round(punishment, 4)
        if notes:
            payload["notes"] = notes
        return payload

    def _build_runtime_tick_artifacts(
        self,
        *,
        run_id: str,
        tick_id: str,
        tick_index: int,
        runtime_tick_index: int,
        input_preview: str,
        runtime_tick: dict[str, Any],
        sandbox_result: dict[str, Any],
        runtime_action_effects: dict[str, Any],
        action_feedback: dict[str, Any],
        teacher_review: dict[str, Any],
        teacher_feedback: dict[str, Any],
        input_item: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        packet = runtime_tick["sensor_packet"]
        competition_packet = runtime_tick["competition_packet"]
        pool_result = runtime_tick["pool_result_external"]
        r_state = runtime_tick["r_state"]
        a_focus = runtime_tick["a_focus"]
        state_pool_summary = runtime_tick["state_pool_summary"]
        state_pool_sidecar = runtime_tick["state_pool_sidecar"]
        state_top = state_pool_summary.get("top", [])
        bn_list = runtime_tick["bn_list"]
        c_i_list = runtime_tick["c_i_list"]
        c_star = runtime_tick["c_star"]
        rules_result = runtime_tick["rules_result"]
        short_term_snapshot = runtime_tick["short_term_snapshot"]
        image_packet = runtime_tick.get("image_packet", {}) or {}
        audio_packet = runtime_tick.get("audio_packet", {}) or {}
        pending_feedback_breakdown = dict(runtime_tick.get("pending_feedback_breakdown", {}) or {})
        fired_rules = list(rules_result.get("rules_fired", []) or [])
        emotion_channels = dict(rules_result.get("emotion_channels", {}) or {})
        action_drives = list(rules_result.get("action_drives", []) or [])
        tuner_result = dict(rules_result.get("tuner_result", {}) or {})
        tuner_matched_profiles = list(tuner_result.get("matched_profiles", []) or [])
        applied_tuner_adjustments = list(runtime_tick.get("applied_tuner_adjustments", []) or [])
        runtime_controls = dict(runtime_tick.get("runtime_controls", {}) or {})
        logic_feedback = dict(runtime_tick.get("logic_feedback", {}) or {})
        action_learning_bias_summary = list(runtime_tick.get("action_learning_bias_summary", []) or [])
        action_learning_context_bias_summary = list(runtime_tick.get("action_learning_context_bias_summary", []) or [])
        tuner_learning_summary = dict(runtime_tick.get("tuner_learning_summary", {}) or {})
        teacher_review = dict(teacher_review or {})
        teacher_feedback = dict(teacher_feedback or {})
        selected_actions = list(sandbox_result.get("selected_actions", []) or [])
        stream_source_meta = dict(input_item.get("stream_source_meta", {}) or {}) if isinstance(input_item.get("stream_source_meta"), dict) else {}
        stream_frame_meta = dict(input_item.get("stream_frame_meta", {}) or {}) if isinstance(input_item.get("stream_frame_meta"), dict) else {}
        stream_chunk_meta = dict(input_item.get("stream_chunk_meta", {}) or {}) if isinstance(input_item.get("stream_chunk_meta"), dict) else {}

        summary = {
            "schema_id": "tick_summary/v1",
            "schema_version": "1.0",
            "run_id": run_id,
            "tick_index": tick_index,
            "runtime_tick_index": runtime_tick_index,
            "tick_id": tick_id,
            "status": "ok",
            "generated_at_ms": now_ms(),
            "input_preview": input_preview,
            "state_top": [{"sa_label": item.get("sa_label", ""), "energy": item.get("energy", 0.0)} for item in state_top[:8]],
            "r_state_preview": list(r_state.get("merged_preview", []))[:12],
            "a_focus_preview": list(a_focus.get("focus_units", [])),
            "sensor_summary": {
                "budget_used": packet.get("budget_used", 0),
                "total_units": packet.get("total_units", 0),
                "suppressed_count": ((packet.get("fatigue_summary") or {}).get("suppressed_count", 0)),
            },
            "multimodal_summary": {
                "has_image": bool(image_packet),
                "has_audio": bool(audio_packet),
                "image_patch_budget_used": int(image_packet.get("budget_used", 0) or 0),
                "image_total_patch_count": int(image_packet.get("total_patch_count", 0) or 0),
                "image_reconstruction_patch_budget": int(image_packet.get("reconstruction_patch_budget", 0) or 0),
                "image_reconstruction_cell_count": int(((image_packet.get("reconstruction_grid", {}) or {}).get("cell_count", 0) or 0)),
                "audio_window_budget_used": int(audio_packet.get("budget_used", 0) or 0),
                "screen_capture": dict((input_item.get("_screenshot_meta", {}) or {}).get("meta", {}) or {}),
                "stream_source": stream_source_meta,
                "stream_frame": stream_frame_meta,
                "stream_chunk": stream_chunk_meta,
            },
            "competition_summary": runtime_tick.get("competition_summary", {}),
            "bn_preview": [
                {
                    "memory_id": item.get("memory_id", ""),
                    "score": item.get("score", 0.0),
                    "text": item.get("text", ""),
                    "candidate_sources": item.get("candidate_sources", []),
                    "score_breakdown": item.get("score_breakdown", {}),
                }
                for item in bn_list[:5]
            ],
            "c_star_preview": [{"sa_label": item.get("sa_label", ""), "energy": item.get("energy", 0.0)} for item in (c_star.get("items", []) or [])[:8]],
            "rules_preview": {
                "rules_fired": [item.get("rule_id", "") for item in fired_rules[:6]],
                "emotion_channels": emotion_channels,
                "sandbox_actions": selected_actions,
                "runtime_action_effects": runtime_action_effects,
                "action_feedback": action_feedback,
                "teacher_review": teacher_review,
                "teacher_feedback": teacher_feedback,
                "feedback_breakdown": pending_feedback_breakdown,
                "rule_fired_count": len(fired_rules),
                "action_drive_count": len(action_drives),
                "sandbox_action_count": len(selected_actions),
                "tuner_matched_count": len(tuner_matched_profiles),
                "tuner_adjustment_count": len(applied_tuner_adjustments),
                "action_learning_bias_count": len(action_learning_bias_summary),
                "action_learning_context_bias_count": len(action_learning_context_bias_summary),
                "tuner_learning_offset_count": len(tuner_learning_summary.get("applied_offsets", []) or []),
            },
            "bn_candidate_source_histogram": self._candidate_source_histogram(bn_list),
            "rule_log_preview": list(rules_result.get("rule_logs", []) or [])[:8],
            "tuner_preview": tuner_result,
            "applied_tuner_adjustments": applied_tuner_adjustments,
            "runtime_controls": runtime_controls,
            "logic_feedback": logic_feedback,
            "action_feedback": action_feedback,
            "teacher_review": teacher_review,
            "teacher_feedback": teacher_feedback,
            "feedback_breakdown": pending_feedback_breakdown,
            "action_learning_bias_summary": action_learning_bias_summary,
            "action_learning_context_bias_summary": action_learning_context_bias_summary,
            "tuner_learning_summary": tuner_learning_summary,
            "memory_index_summary": runtime_tick.get("memory_index_summary", {}),
            "short_term_preview": short_term_snapshot[-3:],
            "state_pool_summary": state_pool_summary,
            "state_pool_sidecar_summary": {
                "anchor_top": [item.get("display_text", "") for item in ((state_pool_summary.get("anchor_summary") or {}).get("top", []) or [])[:6]],
                "residual_top": [item.get("display_text", "") for item in ((state_pool_summary.get("residual_summary") or {}).get("top", []) or [])[:6]],
            },
            "r_state_heads": [{"head_id": head.get("head_id", ""), "preview": [item.get("display_text", "") for item in head.get("items", [])[:6]]} for head in r_state.get("heads", [])],
            "notes": [
                "本 tick 已走通：感受器输入 -> SA 竞争 -> 状态池 -> R_state -> Bn -> C_i -> C* -> rules -> sandbox -> A_focus。",
                "当前已具备工程化向量/ANN/时空联合索引与受限电脑执行骨架；更大规模外部部署与更强设备级控制仍是后续增强目标。",
            ],
        }
        validate_or_raise(summary, load_schema("tick_summary.schema.json"), label="tick_summary")

        post_action_attention_modulation = dict((runtime_action_effects.get("attention_modulation", {}) or {}))
        post_action_effective_controls = dict(runtime_tick.get("effective_attention_controls", {}) or {})
        if not post_action_effective_controls:
            post_action_effective_controls = dict(runtime_tick.get("runtime_controls", {}) or {})
        for key, value in dict(post_action_attention_modulation.get("modulated_controls", {}) or {}).items():
            if not key:
                continue
            try:
                post_action_effective_controls[str(key)] = round(
                    max(float(post_action_effective_controls.get(key, value) or value), float(value)),
                    4,
                )
            except Exception:
                continue

        sidecar = {
            "schema_id": "tick_sidecar/v1",
            "schema_version": "1.1",
            "run_id": run_id,
            "tick_index": tick_index,
            "runtime_tick_index": runtime_tick_index,
            "sensor_packet": packet,
            "image_packet": image_packet,
            "audio_packet": audio_packet,
            "competition_packet": competition_packet,
            "r_state": r_state,
            "bn_list": bn_list,
            "c_i_list": c_i_list,
            "c_star": c_star,
            "rules_result": rules_result,
            "sandbox_result": sandbox_result,
            "runtime_action_effects": runtime_action_effects,
            "action_feedback": action_feedback,
            "teacher_review": teacher_review,
            "teacher_feedback": teacher_feedback,
            "feedback_breakdown": pending_feedback_breakdown,
            "input_item": self._serialize_input_item(input_item),
            "a_focus": a_focus,
            "short_term_snapshot": short_term_snapshot,
            "state_pool_snapshot": state_pool_summary,
            "state_pool_sidecar": state_pool_sidecar,
            "runtime_controls": runtime_controls,
            "effective_attention_controls": runtime_tick.get("effective_attention_controls", {}),
            "final_focus_attention_controls": runtime_tick.get("final_focus_attention_controls", {}),
            "attention_modulation_state": runtime_tick.get("attention_modulation_state", {}),
            "post_action_effective_attention_controls": post_action_effective_controls,
            "post_action_attention_modulation_state": post_action_attention_modulation,
            "applied_tuner_adjustments": applied_tuner_adjustments,
            "logic_feedback": logic_feedback,
            "action_learning_bias_summary": action_learning_bias_summary,
            "action_learning_context_bias_summary": action_learning_context_bias_summary,
            "tuner_learning_summary": tuner_learning_summary,
            "pool_result": pool_result,
            "pool_result_predict": runtime_tick.get("pool_result_predict", {}),
            "pool_result_rules": runtime_tick.get("pool_result_rules", {}),
            "focus_memory": runtime_tick.get("focus_memory", {}),
            "exact_memory": runtime_tick.get("exact_memory", {}),
            "memory_count": runtime_tick.get("memory_count", 0),
            "memory_index_summary": runtime_tick.get("memory_index_summary", {}),
            "stream_source_meta": stream_source_meta,
        }
        sidecar["bn_list"] = self._compact_bn_list_for_storage(sidecar.get("bn_list", []))
        sidecar["c_i_list"] = self._compact_c_i_list_for_storage(sidecar.get("c_i_list", []))

        metrics = {
            "schema_id": "tick_metrics/v1",
            "schema_version": "1.0",
            "run_id": run_id,
            "tick_index": tick_index,
            "runtime_tick_index": runtime_tick_index,
            "logic_ms": 0.0,
            "live_payload_bytes": len(json.dumps({"input_preview": summary["input_preview"], "a_focus_preview": summary["a_focus_preview"]}, ensure_ascii=False).encode("utf-8")),
            "state_top_count": len(summary["state_top"]),
            "r_state_count": len(summary["r_state_preview"]),
            "a_focus_count": len(summary["a_focus_preview"]),
            "state_pool_size": int((summary.get("state_pool_summary") or {}).get("state_pool_size", 0)),
            "state_pool_anchor_count": int(((summary.get("state_pool_summary") or {}).get("anchor_summary") or {}).get("count", 0)),
            "state_pool_residual_count": int(((summary.get("state_pool_summary") or {}).get("residual_summary") or {}).get("count", 0)),
            "bn_count": len(bn_list),
            "c_star_count": len((c_star.get("items", []) or [])),
            "text_budget_used": int((summary.get("sensor_summary") or {}).get("budget_used", 0)),
            "vision_budget_used": int((summary.get("multimodal_summary") or {}).get("image_patch_budget_used", 0)),
            "vision_total_patch_count": int((summary.get("multimodal_summary") or {}).get("image_total_patch_count", 0)),
            "vision_reconstruction_cell_count": int((summary.get("multimodal_summary") or {}).get("image_reconstruction_cell_count", 0)),
            "audio_budget_used": int((summary.get("multimodal_summary") or {}).get("audio_window_budget_used", 0)),
            "stream_item_index": int(stream_source_meta.get("item_index", -1) or -1) if stream_source_meta else -1,
            "stream_total_items": int(stream_source_meta.get("total_items", 0) or 0) if stream_source_meta else 0,
            "rule_fired_count": len(fired_rules),
            "action_drive_count": len(action_drives),
            "sandbox_action_count": len(selected_actions),
            "tuner_matched_count": len(tuner_matched_profiles),
            "tuner_adjustment_count": len(applied_tuner_adjustments),
            "action_learning_bias_count": len(action_learning_bias_summary),
            "tuner_learning_offset_count": len(tuner_learning_summary.get("applied_offsets", []) or []),
            "feedback_reward": float((pending_feedback_breakdown.get("reward", 0.0) or 0.0)),
            "feedback_punishment": float((pending_feedback_breakdown.get("punishment", 0.0) or 0.0)),
            "feedback_intrinsic_reward": self._feedback_source_metric(pending_feedback_breakdown, "intrinsic", "reward"),
            "feedback_intrinsic_punishment": self._feedback_source_metric(pending_feedback_breakdown, "intrinsic", "punishment"),
            "runtime_stage_timing_ms": dict(runtime_tick.get("runtime_stage_timing_ms", {}) or {}),
        }
        return summary, sidecar, metrics

    def _candidate_source_histogram(self, bn_list: list[dict[str, Any]]) -> dict[str, int]:
        hist: dict[str, int] = {}
        for item in bn_list:
            for source in item.get("candidate_sources", []) or []:
                clean = str(source or "")
                if not clean:
                    continue
                hist[clean] = int(hist.get(clean, 0) or 0) + 1
        return hist

    def _update_run_caches(self, *, run_dir: Path, summary: dict[str, Any], metrics: dict[str, Any]) -> None:
        current_rollup = read_json(run_dir / "live" / "run_rollup.json", default=empty_rollup(run_id=str(summary.get("run_id", "") or "")))
        next_rollup = update_rollup(current_rollup, summary=summary, metrics=metrics, series_tail_limit=min(192, self.config.observatory_tick_list_limit))
        write_json(run_dir / "live" / "run_rollup.json", next_rollup)

        tick_cache_path = run_dir / "live" / "tick_list.json"
        tick_cache = read_json(tick_cache_path, default={"schema_id": "tick_list_cache/v1", "run_id": str(summary.get("run_id", "") or ""), "ticks": []})
        ticks = [dict(item) for item in (tick_cache.get("ticks", []) or []) if isinstance(item, dict)]
        current_tick_index = int(-1 if summary.get("tick_index", -1) is None else summary.get("tick_index", -1))
        ticks = [item for item in ticks if int(-1 if item.get("tick_index", -1) is None else item.get("tick_index", -1)) != current_tick_index]
        ticks.append(self._build_tick_cache_row(summary))
        ticks.sort(key=lambda item: int(-1 if item.get("tick_index", -1) is None else item.get("tick_index", -1)))
        tick_cache["ticks"] = ticks[-self.config.observatory_tick_list_limit :]
        tick_cache["run_id"] = str(summary.get("run_id", "") or tick_cache.get("run_id", ""))
        write_json(tick_cache_path, tick_cache)

    def _read_tick_chunk_row(self, run_dir: Path, *, kind: str, tick_index: int) -> dict[str, Any]:
        path = chunk_file(run_dir, kind=kind, tick_index=tick_index, chunk_size=self.config.run_chunk_size)
        if not path.exists():
            return {}
        for row in iter_jsonl(path):
            if isinstance(row, dict) and int(row.get("tick_index", -1)) == int(tick_index):
                return row
        return {}

    def _externalized_payload_specs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("image_packet", "vision"),
            ("audio_packet", "audio"),
            ("competition_packet", "competition"),
            ("r_state", "rstate"),
            ("state_pool_sidecar", "pool"),
            ("focus_memory", "focusmem"),
            ("exact_memory", "exactmem"),
        )

    def _build_storage_sidecar(self, sidecar: dict[str, Any], *, run_dir: Path, run_id: str, tick_index: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        storage_row = copy.deepcopy(sidecar)
        externalized_rows: dict[str, dict[str, Any]] = {}
        for field_name, kind in self._externalized_payload_specs():
            payload = storage_row.get(field_name, {})
            if not isinstance(payload, dict) or not payload:
                continue
            externalized_payload = copy.deepcopy(payload)
            if field_name == "image_packet":
                reconstruction_grid = externalized_payload.get("reconstruction_grid", {}) or {}
                cells = reconstruction_grid.get("cells")
                if isinstance(cells, list) and cells:
                    compact_grid = {key: copy.deepcopy(value) for key, value in reconstruction_grid.items() if key != "cells"}
                    compact_grid["cell_count"] = int(compact_grid.get("cell_count", len(cells)) or len(cells))
                    compact_grid["dense_cells_externalized"] = True
                    compact_grid["storage_note"] = "dense_cells_moved_to_vision_chunk"
                    payload["reconstruction_grid"] = compact_grid
            externalized_rows[kind] = self._compact_media_assets_for_storage(run_dir, externalized_payload, category=kind)
            storage_row[field_name] = {
                "schema_id": f"sidecar_ref/{kind}",
                "run_id": run_id,
                "tick_index": int(tick_index),
                "kind": kind,
                "externalized": True,
            }
        return storage_row, externalized_rows

    def _restore_externalized_sidecar_payloads(self, *, run_dir: Path, tick_index: int, sidecar: dict[str, Any]) -> dict[str, Any]:
        hydrated = copy.deepcopy(sidecar)
        for field_name, kind in self._externalized_payload_specs():
            payload = hydrated.get(field_name, {})
            if not isinstance(payload, dict) or not payload or not bool(payload.get("externalized")):
                continue
            restored = self._read_tick_chunk_row(run_dir, kind=kind, tick_index=tick_index)
            if isinstance(restored, dict) and restored:
                hydrated[field_name] = self._restore_media_assets_from_storage(run_dir, restored)
        return hydrated

    def _serialize_input_item(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = {"text": str(item.get("text", "") or ""), "source_type": str(item.get("source_type", "multimodal_input") or "multimodal_input")}
        if item.get("image_bytes") is not None:
            payload["image_bytes_len"] = len(bytes(item.get("image_bytes") or b""))
        if item.get("audio_bytes") is not None:
            payload["audio_bytes_len"] = len(bytes(item.get("audio_bytes") or b""))
        if isinstance(item.get("stream_source_meta"), dict):
            payload["stream_source_meta"] = dict(item.get("stream_source_meta", {}) or {})
        if isinstance(item.get("stream_frame_meta"), dict):
            payload["stream_frame_meta"] = dict(item.get("stream_frame_meta", {}) or {})
        if isinstance(item.get("stream_chunk_meta"), dict):
            payload["stream_chunk_meta"] = dict(item.get("stream_chunk_meta", {}) or {})
        if isinstance(item.get("external_feedback"), dict):
            payload["external_feedback"] = dict(item.get("external_feedback", {}) or {})
        if isinstance(item.get("autonomous_tick_meta"), dict):
            payload["autonomous_tick_meta"] = dict(item.get("autonomous_tick_meta", {}) or {})
        return payload

    def _build_multimodal_preview(self, item: dict[str, Any]) -> str:
        chunks: list[str] = []
        text = str(item.get("text", "") or "")
        if text:
            chunks.append(text[:48])
        if item.get("image_bytes") is not None:
            chunks.append("[image]")
        if item.get("audio_bytes") is not None:
            chunks.append("[audio]")
        if isinstance(item.get("stream_source_meta"), dict):
            meta = dict(item.get("stream_source_meta", {}) or {})
            source_kind = str(meta.get("source_kind", "") or "")
            item_index = int(meta.get("item_index", -1) or -1)
            total_items = int(meta.get("total_items", 0) or 0)
            if source_kind:
                chunks.append(f"[{source_kind}:{item_index + 1}/{total_items or '?'}]")
        return " ".join(chunks) if chunks else "[empty]"

    def _advance_manifest(self, *, run_id: str, run_dir: Path, tick_index: int) -> None:
        manifest = self._read_manifest_raw(run_id)
        manifest["tick_done"] = tick_index + 1
        manifest["latest_tick_index"] = tick_index
        manifest["updated_at_ms"] = now_ms()
        if self._active_stream_source is not None:
            manifest["source_meta"] = self._active_stream_source.status()
        self._write_manifest(run_dir, manifest)

    def _complete_run(
        self,
        *,
        run_id: str,
        run_dir: Path,
        final_tick_index: int,
        completion_status: str = "completed",
        completion_note: str = "",
        completion_details: dict[str, Any] | None = None,
    ) -> None:
        finished_at = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = completion_status
        manifest["finished_at_ms"] = finished_at
        manifest["updated_at_ms"] = finished_at
        if completion_note:
            notes = list(manifest.get("notes", []) or [])
            notes.append(completion_note)
            manifest["notes"] = notes
        if completion_details:
            manifest["completion_details"] = dict(completion_details)
        self._write_manifest(run_dir, manifest)
        append_jsonl(
            run_dir / "system" / "events.jsonl",
            {
                "ts_ms": finished_at,
                "type": "run_completed" if completion_status == "completed" else "run_stopped",
                "run_id": run_id,
                "status": completion_status,
                "note": completion_note,
                "details": dict(completion_details or {}),
            },
        )
        latest_summary = self.get_tick_summary(run_id, final_tick_index)
        latest_metrics = {
            "schema_id": "tick_metrics/v1",
            "schema_version": "1.0",
            "run_id": run_id,
            "tick_index": final_tick_index,
            "logic_ms": 0.0,
            "live_payload_bytes": 0,
            "state_top_count": len(latest_summary.get("state_top", [])),
            "r_state_count": len(latest_summary.get("r_state_preview", [])),
            "a_focus_count": len(latest_summary.get("a_focus_preview", [])),
        }
        live_status = "completed" if completion_status == "completed" else completion_status
        self._publish_live(run_id=run_id, run_dir=run_dir, summary=latest_summary or {"tick_index": final_tick_index, "tick_id": "", "input_preview": "", "a_focus_preview": []}, metrics=latest_metrics, status=live_status, append_to_ring=False)

    def _fail_run(self, *, run_id: str, run_dir: Path, error: str) -> None:
        failed_at = now_ms()
        manifest = self._read_manifest_raw(run_id)
        manifest["status"] = "failed"
        manifest["finished_at_ms"] = failed_at
        manifest["updated_at_ms"] = failed_at
        self._write_manifest(run_dir, manifest)
        append_jsonl(run_dir / "system" / "events.jsonl", {"ts_ms": failed_at, "type": "run_failed", "run_id": run_id, "error": error})

    def _publish_live(
        self,
        *,
        run_id: str,
        run_dir: Path,
        summary: dict[str, Any],
        metrics: dict[str, Any],
        status: str,
        append_to_ring: bool = True,
    ) -> None:
        live_payload = {
            "schema_id": "live_snapshot/v1",
            "status": status,
            "active_run_id": run_id if status == "running" else "",
            "latest_run_id": run_id,
            "latest_tick": {
                "tick_index": summary.get("tick_index", -1),
                "tick_id": summary.get("tick_id", ""),
                "input_preview": summary.get("input_preview", ""),
                "a_focus_preview": summary.get("a_focus_preview", []),
                "logic_ms": metrics.get("logic_ms", 0.0),
            },
            "server_time_ms": now_ms(),
        }
        with self._lock:
            if self._autonomous_session_status:
                live_payload["autonomous_session"] = self._ensure_autonomous_session_status_defaults(copy.deepcopy(self._autonomous_session_status))
            self._latest_live = copy.deepcopy(live_payload)
            ring_row = {
                "tick_index": summary.get("tick_index", -1),
                "tick_id": summary.get("tick_id", ""),
                "input_preview": summary.get("input_preview", ""),
                "a_focus_preview": summary.get("a_focus_preview", []),
            }
            if append_to_ring:
                tick_id = str(ring_row.get("tick_id", "") or "")
                seen_tick_ids = {str(item.get("tick_id", "") or "") for item in self._live_ring}
                if tick_id and tick_id not in seen_tick_ids:
                    self._live_ring.append(ring_row)
            disk_payload = copy.deepcopy(self._latest_live)
            disk_payload["recent_ticks"] = list(self._live_ring)
            write_json(run_dir / "live" / "latest.json", disk_payload)
