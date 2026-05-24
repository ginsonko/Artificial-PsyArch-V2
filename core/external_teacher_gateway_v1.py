# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from observatory_v2.schema_tools import load_schema, validate_or_raise


def _round4(value: float) -> float:
    return round(float(value), 4)


class ExternalTeacherGatewayV1:
    """
    Formal constrained external teacher gateway.

    The external teacher is only allowed to:
    - review AP-generated action candidates
    - return allow / warn / block
    - attach bounded reward / punishment
    - attach bounded explanation / risk tags

    It is not allowed to replace AP cognition, AP prediction, or AP action generation.
    """

    ALLOWED_MODES = {"off", "stub_file", "http_json"}
    ALLOWED_DECISIONS = {"allow", "warn", "block"}
    ALLOWED_RISK_TAGS = {
        "risky_input",
        "unsafe_action",
        "repeated_action",
        "low_confidence",
        "residual_unresolved",
        "typing_without_focus",
        "rate_limited",
        "unknown",
    }

    def __init__(
        self,
        *,
        mode: str = "off",
        stub_response_path: str = "",
        timeout_ms: int = 150,
        max_retries: int = 1,
        retry_backoff_ms: int = 25,
        http_endpoint: str = "",
        http_headers: dict[str, Any] | None = None,
    ) -> None:
        self.mode = self._normalize_mode(mode)
        self.stub_response_path = str(stub_response_path or "").strip()
        self.timeout_ms = max(1, int(timeout_ms))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.http_endpoint = str(http_endpoint or "").strip()
        self.http_headers = self._sanitize_headers(http_headers or {})
        self._providers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "stub_file": self._review_via_stub_file,
            "http_json": self._review_via_http_json,
        }

    def review(self, *, request_payload: dict[str, Any]) -> dict[str, Any]:
        protocol_request = self._build_request_payload(request_payload)
        if self.mode == "off":
            return {
                "ok": False,
                "mode": "off",
                "provider": "off",
                "error": "external_teacher_gateway_disabled",
                "request_digest": protocol_request["request_digest"],
                "transport_audit": self._build_transport_audit(
                    provider="off",
                    attempt_count=0,
                    success=False,
                    transport_result="disabled",
                    transport_error_kind="gateway_disabled",
                ),
            }
        handler = self._providers.get(self.mode)
        if handler is None:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": self.mode,
                "error": f"unsupported_external_teacher_mode:{self.mode}",
                "request_digest": protocol_request["request_digest"],
                "transport_audit": self._build_transport_audit(
                    provider=self.mode,
                    attempt_count=0,
                    success=False,
                    transport_result="unsupported_mode",
                    transport_error_kind="unsupported_mode",
                ),
            }
        response = handler(protocol_request)
        return self._finalize_response(response=response, protocol_request=protocol_request)

    def export_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stub_response_path": self.stub_response_path,
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "retry_backoff_ms": self.retry_backoff_ms,
            "http_endpoint": self.http_endpoint,
            "http_headers": dict(self.http_headers),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        payload = dict(payload or {})
        self.mode = self._normalize_mode(payload.get("mode", self.mode))
        self.stub_response_path = str(payload.get("stub_response_path", self.stub_response_path) or self.stub_response_path)
        self.timeout_ms = max(1, int(payload.get("timeout_ms", self.timeout_ms) or self.timeout_ms))
        self.max_retries = max(1, int(payload.get("max_retries", self.max_retries) or self.max_retries))
        self.retry_backoff_ms = max(0, int(payload.get("retry_backoff_ms", self.retry_backoff_ms) or self.retry_backoff_ms))
        self.http_endpoint = str(payload.get("http_endpoint", self.http_endpoint) or self.http_endpoint).strip()
        self.http_headers = self._sanitize_headers(payload.get("http_headers", self.http_headers) or self.http_headers)

    def _normalize_mode(self, raw: Any) -> str:
        mode = str(raw or "off").strip().lower()
        return mode if mode in self.ALLOWED_MODES else "off"

    def _sanitize_headers(self, headers: dict[str, Any]) -> dict[str, str]:
        clean: dict[str, str] = {}
        for key, value in dict(headers or {}).items():
            hkey = str(key or "").strip()
            if not hkey:
                continue
            clean[hkey[:64]] = str(value or "")[:512]
        return clean

    def _build_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_actions = [dict(item) for item in (payload.get("candidate_actions", []) or []) if isinstance(item, dict)]
        actions = []
        for item in raw_actions[:16]:
            action_id = str(item.get("action_id", "") or "").strip()[:96]
            action_name = str(item.get("action_name", "") or "").strip()[:96] or action_id.replace("action::", "")
            actions.append(
                {
                    "action_id": action_id,
                    "action_name": action_name,
                    "drive": _round4(max(0.0, min(1.5, float(item.get("drive", 0.0) or 0.0)))),
                }
            )
        state_summary = dict(payload.get("state_summary", {}) or {})
        request_digest = {
            "tick_index": int(payload.get("tick_index", -1) or -1),
            "candidate_action_count": len(actions),
            "top_actions": [
                {
                    "action_id": str(item.get("action_id", "") or ""),
                    "action_name": str(item.get("action_name", "") or ""),
                    "drive": _round4(float(item.get("drive", 0.0) or 0.0)),
                }
                for item in actions[:4]
            ],
            "focus_preview": [str(item or "")[:96] for item in (payload.get("focus_preview", []) or [])[:6]],
            "bn_preview_ids": [str(item or "")[:96] for item in (payload.get("bn_preview_ids", []) or [])[:6]],
        }
        protocol_request = {
            "schema_id": "external_teacher_request/v1",
            "schema_version": "1.0",
            "tick_index": int(payload.get("tick_index", -1) or -1),
            "candidate_actions": actions,
            "focus_preview": request_digest["focus_preview"],
            "bn_preview_ids": request_digest["bn_preview_ids"],
            "state_summary": {
                "state_pool_size": int(state_summary.get("state_pool_size", 0) or 0),
                "recent_external_count": int(state_summary.get("recent_external_count", 0) or 0),
                "verbatim_chars": int(state_summary.get("verbatim_chars", 0) or 0),
                "anchor_count": int(((state_summary.get("anchor_summary") or {}).get("count", 0)) or 0),
                "residual_count": int(((state_summary.get("residual_summary") or {}).get("count", 0)) or 0),
                "residual_total_unresolved_mass": _round4(
                    float(((state_summary.get("residual_summary") or {}).get("total_unresolved_mass", 0.0)) or 0.0)
                ),
            },
            "request_digest": request_digest,
        }
        validate_or_raise(protocol_request, load_schema("external_teacher_request.schema.json"), label="external_teacher_request")
        return protocol_request

    def _finalize_response(self, *, response: dict[str, Any], protocol_request: dict[str, Any]) -> dict[str, Any]:
        raw = dict(response or {})
        ok = bool(raw.get("ok", False))
        mode = self._normalize_mode(raw.get("mode", self.mode))
        provider = str(raw.get("provider", mode) or mode).strip()[:64]
        reviewer = str(raw.get("reviewer", provider or mode) or provider or mode).strip()[:96]
        path = str(raw.get("path", "") or "")[:512]
        request_digest = dict(raw.get("request_digest", {}) or protocol_request.get("request_digest", {}) or {})
        error = str(raw.get("error", "") or "").strip()[:256]
        transport_audit = self._sanitize_transport_audit(raw.get("transport_audit", {}) or {})
        decisions = []
        for item in raw.get("decisions", []) or []:
            if not isinstance(item, dict):
                continue
            decisions.append(self._sanitize_decision(item))
        payload = {
            "schema_id": "external_teacher_response/v1",
            "schema_version": "1.0",
            "ok": ok,
            "mode": mode,
            "provider": provider,
            "reviewer": reviewer,
            "path": path,
            "request_digest": request_digest,
            "decisions": decisions[:16],
            "error": error,
            "transport_audit": transport_audit,
        }
        validate_or_raise(payload, load_schema("external_teacher_response.schema.json"), label="external_teacher_response")
        return payload

    def _review_via_stub_file(self, protocol_request: dict[str, Any]) -> dict[str, Any]:
        if not self.stub_response_path:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "stub_file",
                "reviewer": "stub_file",
                "error": "external_teacher_stub_path_missing",
                "transport_audit": self._build_transport_audit(
                    provider="stub_file",
                    attempt_count=1,
                    success=False,
                    transport_error_kind="path_missing",
                ),
            }
        path = Path(self.stub_response_path)
        if not path.exists():
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "stub_file",
                "reviewer": "stub_file",
                "error": "external_teacher_stub_not_found",
                "path": str(path),
                "transport_audit": self._build_transport_audit(
                    provider="stub_file",
                    attempt_count=1,
                    success=False,
                    transport_error_kind="not_found",
                ),
            }
        try:
            started = time.perf_counter()
            payload = json.loads(path.read_text(encoding="utf-8"))
            duration_ms = self._elapsed_ms(started)
        except Exception as exc:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "stub_file",
                "reviewer": "stub_file",
                "error": f"external_teacher_stub_decode_failed:{exc}",
                "path": str(path),
                "transport_audit": self._build_transport_audit(
                    provider="stub_file",
                    attempt_count=1,
                    success=False,
                    duration_ms=self._elapsed_ms(started),
                    transport_error_kind="decode_failed",
                ),
            }
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "stub_file",
                "reviewer": "stub_file",
                "error": "external_teacher_stub_invalid_root",
                "path": str(path),
                "transport_audit": self._build_transport_audit(
                    provider="stub_file",
                    attempt_count=1,
                    success=False,
                    duration_ms=duration_ms,
                    transport_error_kind="invalid_root",
                ),
            }
        if payload.get("schema_id") == "external_teacher_response/v1":
            payload = dict(payload)
            payload.setdefault("mode", self.mode)
            payload.setdefault("provider", "stub_file")
            payload.setdefault("reviewer", str(payload.get("reviewer", "stub_file") or "stub_file"))
            payload.setdefault("path", str(path))
            payload["transport_audit"] = self._build_transport_audit(
                provider="stub_file",
                attempt_count=1,
                success=True,
                duration_ms=duration_ms,
                transport_result="ok",
            )
            return payload
        decisions = []
        for raw in payload.get("decisions", []) or []:
            if not isinstance(raw, dict):
                continue
            decisions.append(self._sanitize_decision(raw))
        return {
            "ok": True,
            "mode": self.mode,
            "provider": "stub_file",
            "reviewer": str(payload.get("reviewer", "stub_file") or "stub_file"),
            "path": str(path),
            "request_digest": dict(protocol_request.get("request_digest", {}) or {}),
            "decisions": decisions,
            "error": "",
            "transport_audit": self._build_transport_audit(
                provider="stub_file",
                attempt_count=1,
                success=True,
                duration_ms=duration_ms,
                transport_result="ok",
            ),
        }

    def _review_via_http_json(self, protocol_request: dict[str, Any]) -> dict[str, Any]:
        endpoint = str(self.http_endpoint or "").strip()
        if not endpoint:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "http_json",
                "reviewer": "http_json",
                "error": "external_teacher_http_endpoint_missing",
                "transport_audit": self._build_transport_audit(
                    provider="http_json",
                    attempt_count=1,
                    success=False,
                    transport_error_kind="endpoint_missing",
                ),
            }
        body = json.dumps(protocol_request, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
        headers.update(self.http_headers)
        timeout_sec = max(0.05, float(self.timeout_ms) / 1000.0)
        last_error = ""
        last_status_code = 0
        last_error_kind = ""
        total_started = time.perf_counter()
        raw_bytes = b""
        attempt_count = 0
        for attempt_index in range(self.max_retries):
            attempt_count = attempt_index + 1
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    raw_bytes = response.read()
                    last_status_code = int(getattr(response, "status", 200) or 200)
                last_error = ""
                last_error_kind = ""
                break
            except urllib.error.HTTPError as exc:
                last_status_code = int(exc.code)
                last_error_kind = "http_status"
                last_error = f"external_teacher_http_status:{int(exc.code)}"
            except urllib.error.URLError as exc:
                last_error_kind = "url_error"
                last_error = f"external_teacher_http_failed:{exc}"
            except TimeoutError as exc:
                last_error_kind = "timeout"
                last_error = f"external_teacher_http_failed:{exc}"
            except Exception as exc:
                last_error_kind = exc.__class__.__name__.lower()[:64] or "http_failed"
                last_error = f"external_teacher_http_failed:{exc}"
            if attempt_count < self.max_retries and self.retry_backoff_ms > 0:
                time.sleep(self.retry_backoff_ms / 1000.0)
        duration_ms = self._elapsed_ms(total_started)
        if last_error:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "http_json",
                "reviewer": "http_json",
                "error": last_error,
                "transport_audit": self._build_transport_audit(
                    provider="http_json",
                    attempt_count=attempt_count,
                    success=False,
                    duration_ms=duration_ms,
                    status_code=last_status_code,
                    transport_error_kind=last_error_kind or "http_failed",
                    transport_result="error",
                ),
            }
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "http_json",
                "reviewer": "http_json",
                "error": f"external_teacher_http_decode_failed:{exc}",
                "transport_audit": self._build_transport_audit(
                    provider="http_json",
                    attempt_count=attempt_count,
                    success=False,
                    duration_ms=duration_ms,
                    status_code=last_status_code,
                    transport_error_kind="decode_failed",
                    transport_result="decode_failed",
                ),
            }
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "mode": self.mode,
                "provider": "http_json",
                "reviewer": "http_json",
                "error": "external_teacher_http_invalid_root",
                "transport_audit": self._build_transport_audit(
                    provider="http_json",
                    attempt_count=attempt_count,
                    success=False,
                    duration_ms=duration_ms,
                    status_code=last_status_code,
                    transport_error_kind="invalid_root",
                    transport_result="invalid_root",
                ),
            }
        if payload.get("schema_id") == "external_teacher_response/v1":
            payload = dict(payload)
            payload.setdefault("mode", self.mode)
            payload.setdefault("provider", "http_json")
            payload.setdefault("reviewer", str(payload.get("reviewer", "http_json") or "http_json"))
            payload.setdefault("request_digest", dict(protocol_request.get("request_digest", {}) or {}))
            payload["transport_audit"] = self._build_transport_audit(
                provider="http_json",
                attempt_count=attempt_count,
                success=True,
                duration_ms=duration_ms,
                status_code=last_status_code,
                transport_result="ok",
            )
            return payload
        decisions = []
        for raw in payload.get("decisions", []) or []:
            if not isinstance(raw, dict):
                continue
            decisions.append(self._sanitize_decision(raw))
        return {
            "ok": bool(payload.get("ok", True)),
            "mode": self.mode,
            "provider": "http_json",
            "reviewer": str(payload.get("reviewer", "http_json") or "http_json"),
            "request_digest": dict(protocol_request.get("request_digest", {}) or {}),
            "decisions": decisions,
            "error": str(payload.get("error", "") or ""),
            "transport_audit": self._build_transport_audit(
                provider="http_json",
                attempt_count=attempt_count,
                success=bool(payload.get("ok", True)),
                duration_ms=duration_ms,
                status_code=last_status_code,
                transport_result="ok",
            ),
        }

    def _sanitize_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision = str(payload.get("decision", "allow") or "allow").strip().lower()
        if decision not in self.ALLOWED_DECISIONS:
            decision = "allow"
        risk_tags = []
        for raw in payload.get("risk_tags", []) or []:
            clean = str(raw or "").strip()
            if not clean:
                continue
            risk_tags.append(clean if clean in self.ALLOWED_RISK_TAGS else "unknown")
        return {
            "decision": decision,
            "target_action_id": str(payload.get("target_action_id", "") or "").strip()[:96],
            "target_action_name": str(payload.get("target_action_name", "") or "").strip()[:96],
            "confidence": _round4(max(0.0, min(1.0, float(payload.get("confidence", 0.5) or 0.5)))),
            "reward": _round4(max(0.0, min(2.0, float(payload.get("reward", 0.0) or 0.0)))),
            "punishment": _round4(max(0.0, min(2.0, float(payload.get("punishment", 0.0) or 0.0)))),
            "explanation": str(payload.get("explanation", "") or "")[:400],
            "warning_code": str(payload.get("warning_code", "") or "").strip()[:96],
            "risk_tags": risk_tags[:8],
        }

    def _sanitize_transport_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload or {})
        return {
            "provider": str(data.get("provider", "") or "")[:64],
            "attempt_count": max(0, int(data.get("attempt_count", 0) or 0)),
            "duration_ms": _round4(max(0.0, float(data.get("duration_ms", 0.0) or 0.0))),
            "status_code": max(0, int(data.get("status_code", 0) or 0)),
            "success": bool(data.get("success", False)),
            "transport_result": str(data.get("transport_result", "") or "")[:64],
            "transport_error_kind": str(data.get("transport_error_kind", "") or "")[:64],
        }

    def _build_transport_audit(
        self,
        *,
        provider: str,
        attempt_count: int,
        success: bool,
        duration_ms: float = 0.0,
        status_code: int = 0,
        transport_result: str = "",
        transport_error_kind: str = "",
    ) -> dict[str, Any]:
        return self._sanitize_transport_audit(
            {
                "provider": provider,
                "attempt_count": attempt_count,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "success": success,
                "transport_result": transport_result,
                "transport_error_kind": transport_error_kind,
            }
        )

    def _elapsed_ms(self, started: float) -> float:
        return _round4(max(0.0, (time.perf_counter() - started) * 1000.0))
