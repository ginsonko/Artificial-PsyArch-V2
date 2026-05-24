# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _decision_for_action(action: dict[str, Any], *, warn_drive: float, block_drive: float) -> dict[str, Any]:
    action_id = str(action.get("action_id", "") or "")
    action_name = str(action.get("action_name", "") or "")
    drive = float(action.get("drive", 0.0) or 0.0)
    risky = action_name in {"click", "double_click", "type_text", "press_key", "scroll"}
    if risky and drive >= block_drive:
        decision = "block"
        punishment = 0.18
        warning_code = f"stub_block_{action_name}"
        tags = ["unsafe_action", "stub_teacher"]
        explanation = f"stub teacher blocks high-drive risky action: {action_name}"
    elif risky and drive >= warn_drive:
        decision = "warn"
        punishment = 0.08
        warning_code = f"stub_warn_{action_name}"
        tags = ["risky_action", "stub_teacher"]
        explanation = f"stub teacher warns about risky action: {action_name}"
    else:
        decision = "allow"
        punishment = 0.0
        warning_code = ""
        tags = ["stub_teacher"]
        explanation = f"stub teacher allows action: {action_name or action_id}"
    return {
        "decision": decision,
        "target_action_id": action_id,
        "target_action_name": action_name,
        "confidence": 0.82,
        "reward": 0.02 if decision == "allow" else 0.0,
        "punishment": punishment,
        "explanation": explanation,
        "warning_code": warning_code,
        "risk_tags": tags,
    }


def build_response(request_payload: dict[str, Any], *, warn_drive: float, block_drive: float) -> dict[str, Any]:
    actions = [dict(item) for item in (request_payload.get("candidate_actions", []) or []) if isinstance(item, dict)]
    return {
        "schema_id": "external_teacher_response/v1",
        "schema_version": "1.0",
        "ok": True,
        "mode": "http_json",
        "provider": "local_stub_http",
        "reviewer": "local_stub_http_teacher",
        "path": "/teacher",
        "request_digest": dict(request_payload.get("request_digest", {}) or {}),
        "decisions": [_decision_for_action(action, warn_drive=warn_drive, block_drive=block_drive) for action in actions],
        "error": "",
    }


def make_handler(*, warn_drive: float, block_drive: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/api/health"}:
                self._send_json({"ok": True, "service": "external_teacher_stub_server", "status": "ready"})
                return
            self._send_json({"ok": False, "error": "not_found", "path": self.path}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/", "/teacher", "/api/teacher"}:
                self._send_json({"ok": False, "error": "not_found", "path": self.path}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                request_payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception as exc:
                self._send_json({"ok": False, "error": f"invalid_json: {exc}"}, status=400)
                return
            if not isinstance(request_payload, dict):
                self._send_json({"ok": False, "error": "request must be an object"}, status=400)
                return
            self._send_json(build_response(request_payload, warn_drive=warn_drive, block_drive=block_drive))

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="AP 二期 external teacher http_json 本地验收 stub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--warn-drive", type=float, default=0.55)
    parser.add_argument("--block-drive", type=float, default=0.9)
    args = parser.parse_args()
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(warn_drive=float(args.warn_drive), block_drive=float(args.block_drive)))
    host, port = server.server_address[:2]
    print(f"external teacher stub server ready: http://{host}:{port}/teacher", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
