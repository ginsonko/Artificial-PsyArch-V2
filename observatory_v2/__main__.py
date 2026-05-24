# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from .app import ObservatoryV2App
from .config import load_config
from .dataset_runner import run_dataset_file
from .web import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AP 二期最小观测台")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="启动本地观测台 Web 服务")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--no-browser", action="store_true")

    run_demo = sub.add_parser("run-demo", help="执行一次最小演示运行")
    run_demo.add_argument("--ticks", type=int, default=None)
    run_demo.add_argument("--interval-ms", type=int, default=None)
    run_demo.add_argument("--label", default="Phase1 命令行演示运行")

    run_text = sub.add_parser("run-text", help="执行一次文本输入最小闭环运行")
    run_text.add_argument("--text", action="append", dest="texts", default=[], help="按 tick 输入的一条文本，可重复传入")
    run_text.add_argument("--label", default="Phase2 命令行文本最小闭环运行")
    run_text.add_argument("--interval-ms", type=int, default=0)
    run_text.add_argument("--reset-runtime", action="store_true")

    run_multimodal = sub.add_parser("run-multimodal", help="执行一次统一多模态运行")
    run_multimodal.add_argument("--text", action="append", dest="texts", default=[], help="每个 tick 的文本输入")
    run_multimodal.add_argument("--image", action="append", dest="images", default=[], help="每个 tick 的图片路径")
    run_multimodal.add_argument("--audio", action="append", dest="audios", default=[], help="每个 tick 的音频路径")
    run_multimodal.add_argument("--label", default="Phase11 命令行多模态运行")
    run_multimodal.add_argument("--interval-ms", type=int, default=0)
    run_multimodal.add_argument("--reset-runtime", action="store_true")

    run_dataset = sub.add_parser("run-dataset", help="按数据集描述执行一个或多个工程化 run")
    run_dataset.add_argument("--dataset", required=True, help="JSON dataset path")
    run_dataset.add_argument("--label", default="Phase10 命令行批量实验")
    run_dataset.add_argument("--outputs-root", default="")
    run_dataset.add_argument("--timeout-sec", type=float, default=600.0)

    run_screen = sub.add_parser("run-screen", help="执行一次截图感知运行")
    run_screen.add_argument("--ticks", type=int, default=1)
    run_screen.add_argument("--text", default="")
    run_screen.add_argument("--label", default="Phase17 命令行截图感知运行")
    run_screen.add_argument("--interval-ms", type=int, default=0)
    run_screen.add_argument("--reset-runtime", action="store_true")

    run_audio_stream = sub.add_parser("run-audio-stream", help="执行一次连续音频流运行")
    run_audio_stream.add_argument("--audio", required=True, help="长音频 wav 路径")
    run_audio_stream.add_argument("--text-prefix", default="")
    run_audio_stream.add_argument("--tick-window-ms", type=int, default=0)
    run_audio_stream.add_argument("--label", default="Phase12 命令行连续音频流运行")
    run_audio_stream.add_argument("--interval-ms", type=int, default=0)
    run_audio_stream.add_argument("--reset-runtime", action="store_true")

    run_image_stream = sub.add_parser("run-image-stream", help="执行一次连续图像流运行")
    run_image_stream.add_argument("--frame", action="append", dest="frames", default=[], help="逐帧图片路径，可重复")
    run_image_stream.add_argument("--strip-image", default="", help="竖向拼接帧图路径")
    run_image_stream.add_argument("--frame-count", type=int, default=1)
    run_image_stream.add_argument("--text-prefix", default="")
    run_image_stream.add_argument("--label", default="Phase11 命令行连续图像流运行")
    run_image_stream.add_argument("--interval-ms", type=int, default=0)
    run_image_stream.add_argument("--reset-runtime", action="store_true")

    run_video_stream = sub.add_parser("run-video-stream", help="执行一次连续视频流运行")
    run_video_stream.add_argument("--video", required=True, help="视频文件路径")
    run_video_stream.add_argument("--text-prefix", default="")
    run_video_stream.add_argument("--tick-fps", type=float, default=0.0, help="每 tick 抽样频率，0 表示按 stride")
    run_video_stream.add_argument("--frame-stride", type=int, default=0, help="每隔多少帧抽一帧，0 表示自动")
    run_video_stream.add_argument("--max-frames", type=int, default=0, help="最多取多少帧，0 表示不限")
    run_video_stream.add_argument("--label", default="Phase20 命令行连续视频流运行")
    run_video_stream.add_argument("--interval-ms", type=int, default=0)
    run_video_stream.add_argument("--reset-runtime", action="store_true")

    run_webcam_stream = sub.add_parser("run-webcam-stream", help="执行一次 webcam 实时流运行")
    run_webcam_stream.add_argument("--text-prefix", default="")
    run_webcam_stream.add_argument("--max-frames", type=int, default=0)
    run_webcam_stream.add_argument("--device-index", type=int, default=0)
    run_webcam_stream.add_argument("--frame-width", type=int, default=0)
    run_webcam_stream.add_argument("--frame-height", type=int, default=0)
    run_webcam_stream.add_argument("--label", default="Phase21 命令行 webcam 实时流运行")
    run_webcam_stream.add_argument("--interval-ms", type=int, default=0)
    run_webcam_stream.add_argument("--reset-runtime", action="store_true")

    run_microphone_stream = sub.add_parser("run-microphone-stream", help="执行一次 microphone 实时流运行")
    run_microphone_stream.add_argument("--text-prefix", default="")
    run_microphone_stream.add_argument("--max-windows", type=int, default=0)
    run_microphone_stream.add_argument("--tick-window-ms", type=int, default=0)
    run_microphone_stream.add_argument("--sample-rate", type=int, default=16000)
    run_microphone_stream.add_argument("--channels", type=int, default=1)
    run_microphone_stream.add_argument("--device-index", type=int, default=-1)
    run_microphone_stream.add_argument("--label", default="Phase21 命令行 microphone 实时流运行")
    run_microphone_stream.add_argument("--interval-ms", type=int, default=0)
    run_microphone_stream.add_argument("--reset-runtime", action="store_true")

    run_autonomous = sub.add_parser("run-autonomous", help="执行一次自主电脑循环运行")
    run_autonomous.add_argument("--ticks", type=int, default=4)
    run_autonomous.add_argument("--text-hint", default="")
    run_autonomous.add_argument("--label", default="Phase19 命令行自主电脑循环运行")
    run_autonomous.add_argument("--interval-ms", type=int, default=0)
    run_autonomous.add_argument("--reset-runtime", action="store_true")
    run_autonomous.add_argument("--stop-on-capture-failures", type=int, default=0)
    run_autonomous.add_argument("--stop-on-action-errors", type=int, default=0)
    run_autonomous.add_argument("--stop-on-idle-ticks", type=int, default=0)
    run_autonomous.add_argument("--idle-backoff-ms", type=int, default=0)
    run_autonomous.add_argument("--auto-feedback", choices=["default", "on", "off"], default="default")
    run_autonomous.add_argument("--teacher-mode", choices=["default", "off", "heuristic", "llm_assisted"], default="default")
    run_autonomous.add_argument("--llm-gate-mode", choices=["default", "off", "heuristic", "stub_file"], default="default")
    run_autonomous.add_argument("--external-teacher", choices=["default", "on", "off"], default="default")
    run_autonomous.add_argument("--external-teacher-mode", choices=["default", "off", "stub_file", "http_json"], default="default")
    run_autonomous.add_argument("--external-teacher-stub", default="")
    run_autonomous.add_argument("--external-teacher-fail-open", choices=["default", "on", "off"], default="default")
    run_autonomous.add_argument("--external-teacher-max-retries", type=int, default=None)
    run_autonomous.add_argument("--external-teacher-retry-backoff-ms", type=int, default=None)
    run_autonomous.add_argument("--external-teacher-http-endpoint", default="")

    run_autonomous_session = sub.add_parser("run-autonomous-session", help="启动一个持续自主 session")
    run_autonomous_session.add_argument("--text-hint", default="")
    run_autonomous_session.add_argument("--label", default="Phase20 命令行持续自主 session")
    run_autonomous_session.add_argument("--interval-ms", type=int, default=0)
    run_autonomous_session.add_argument("--reset-runtime", action="store_true")
    run_autonomous_session.add_argument("--max-ticks", type=int, default=0)
    run_autonomous_session.add_argument("--stop-on-capture-failures", type=int, default=0)
    run_autonomous_session.add_argument("--stop-on-action-errors", type=int, default=0)
    run_autonomous_session.add_argument("--stop-on-idle-ticks", type=int, default=0)
    run_autonomous_session.add_argument("--idle-backoff-ms", type=int, default=0)
    run_autonomous_session.add_argument("--auto-feedback", choices=["default", "on", "off"], default="default")
    run_autonomous_session.add_argument("--teacher-mode", choices=["default", "off", "heuristic", "llm_assisted"], default="default")
    run_autonomous_session.add_argument("--llm-gate-mode", choices=["default", "off", "heuristic", "stub_file"], default="default")
    run_autonomous_session.add_argument("--external-teacher", choices=["default", "on", "off"], default="default")
    run_autonomous_session.add_argument("--external-teacher-mode", choices=["default", "off", "stub_file", "http_json"], default="default")
    run_autonomous_session.add_argument("--external-teacher-stub", default="")
    run_autonomous_session.add_argument("--external-teacher-fail-open", choices=["default", "on", "off"], default="default")
    run_autonomous_session.add_argument("--external-teacher-max-retries", type=int, default=None)
    run_autonomous_session.add_argument("--external-teacher-retry-backoff-ms", type=int, default=None)
    run_autonomous_session.add_argument("--external-teacher-http-endpoint", default="")
    run_autonomous_session.add_argument("--wait", action="store_true", help="前台等待 session 结束，适合 max-ticks 验收")
    run_autonomous_session.add_argument("--timeout-sec", type=float, default=120.0, help="--wait 的最长等待秒数")
    run_autonomous_session.add_argument("--server-url", default="", help="连接已有观测台服务，通过 HTTP 启动 session")

    pause_autonomous_session = sub.add_parser("pause-autonomous-session", help="暂停当前持续自主 session")
    pause_autonomous_session.add_argument("--server-url", default="", help="已有观测台服务地址，例如 http://127.0.0.1:8766")
    resume_autonomous_session = sub.add_parser("resume-autonomous-session", help="恢复当前持续自主 session")
    resume_autonomous_session.add_argument("--server-url", default="", help="已有观测台服务地址，例如 http://127.0.0.1:8766")
    stop_autonomous_session = sub.add_parser("stop-autonomous-session", help="停止当前持续自主 session")
    stop_autonomous_session.add_argument("--server-url", default="", help="已有观测台服务地址，例如 http://127.0.0.1:8766")
    recover_autonomous_session = sub.add_parser("recover-autonomous-session", help="恢复一个可恢复的持续自主 session")
    recover_autonomous_session.add_argument("--run-id", default="", help="指定要恢复的 run_id；为空则自动找最近可恢复项")
    recover_autonomous_session.add_argument("--server-url", default="", help="已有观测台服务地址，例如 http://127.0.0.1:8766")
    get_autonomous_session_status = sub.add_parser("autonomous-session-status", help="查看当前持续自主 session 状态")
    get_autonomous_session_status.add_argument("--server-url", default="", help="已有观测台服务地址，例如 http://127.0.0.1:8766")

    export_runtime = sub.add_parser("export-runtime", help="导出当前 runtime")
    export_runtime.add_argument("--out", required=True)

    import_runtime = sub.add_parser("import-runtime", help="导入 runtime checkpoint")
    import_runtime.add_argument("--in", dest="infile", required=True)

    export_memory_bundle = sub.add_parser("export-memory-bundle", help="导出当前记忆部署包")
    export_memory_bundle.add_argument("--out-dir", required=True)

    inspect_memory_bundle = sub.add_parser("inspect-memory-bundle", help="检查记忆部署包内容")
    inspect_memory_bundle.add_argument("--dir", dest="directory", required=True)

    import_memory_bundle = sub.add_parser("import-memory-bundle", help="导入记忆部署包")
    import_memory_bundle.add_argument("--dir", dest="directory", required=True)

    continue_run = sub.add_parser("continue-from-checkpoint", help="导入 checkpoint 后继续文本运行")
    continue_run.add_argument("--in", dest="infile", required=True)
    continue_run.add_argument("--text", action="append", dest="texts", default=[], required=True)
    continue_run.add_argument("--label", default="从 checkpoint 继续")
    continue_run.add_argument("--interval-ms", type=int, default=0)

    forget = sub.add_parser("forget", help="Run memory forgetting / pruning")
    forget.add_argument("--keep-latest", type=int, default=128)
    forget.add_argument("--strategy", choices=["latest_only", "score_prune"], default="latest_only")
    forget.add_argument("--min-reality-weight", type=float, default=0.0)
    forget.add_argument("--min-total-item-energy", type=float, default=0.0)
    forget.add_argument("--protect-memory-kind", action="append", dest="protect_memory_kinds", default=[])
    forget.add_argument("--max-memory-count", type=int, default=0)
    forget.add_argument("--dry-run", action="store_true")

    sub.add_parser("latest-run", help="输出最新 run 的 manifest")
    return parser


def _align_multimodal_inputs(texts: list[str], images: list[str], audios: list[str]) -> list[dict[str, object]]:
    total = max(len(texts), len(images), len(audios), 1)
    items: list[dict[str, object]] = []
    for index in range(total):
        item: dict[str, object] = {"text": texts[index] if index < len(texts) else "", "source_type": "multimodal_input"}
        if index < len(images):
            item["image_bytes"] = Path(images[index]).read_bytes()
        if index < len(audios):
            item["audio_bytes"] = Path(audios[index]).read_bytes()
        items.append(item)
    return items


def _json_request(server_url: str, path: str, *, payload: dict[str, object] | None = None) -> dict[str, object]:
    base = str(server_url or "").strip().rstrip("/")
    if not base:
        raise SystemExit("--server-url is required for remote session control")
    url = f"{base}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8")
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
    parsed = json.loads(data or "{}")
    if not isinstance(parsed, dict):
        return {"ok": False, "error": "remote response is not an object"}
    return parsed


def _print_json(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(text.encode("utf-8", errors="replace") + b"\n")
            buffer.flush()
            return
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def _wait_remote_autonomous_session(server_url: str, *, timeout_sec: float) -> dict[str, object]:
    deadline = time.time() + max(0.1, float(timeout_sec))
    last_status: dict[str, object] = {}
    while time.time() < deadline:
        last_status = _json_request(server_url, "/api/autonomous-session/status")
        if not bool(last_status.get("active", False)):
            return {"ok": True, "status": last_status}
        time.sleep(0.2)
    stop_result = _json_request(server_url, "/api/autonomous-session/stop", payload={})
    for _ in range(50):
        last_status = _json_request(server_url, "/api/autonomous-session/status")
        if not bool(last_status.get("active", False)):
            return {"ok": False, "timeout": True, "stop_result": stop_result, "status": last_status}
        time.sleep(0.1)
    return {"ok": False, "timeout": True, "stop_result": stop_result, "status": last_status}


def _autonomous_session_payload_from_args(args: argparse.Namespace) -> dict[str, object]:
    auto_feedback_enabled = None
    if args.auto_feedback == "on":
        auto_feedback_enabled = True
    elif args.auto_feedback == "off":
        auto_feedback_enabled = False
    teacher_mode = None if args.teacher_mode == "default" else args.teacher_mode
    llm_gate_mode = None if args.llm_gate_mode == "default" else args.llm_gate_mode
    external_teacher_enabled = None
    if args.external_teacher == "on":
        external_teacher_enabled = True
    elif args.external_teacher == "off":
        external_teacher_enabled = False
    external_teacher_fail_open = None
    if args.external_teacher_fail_open == "on":
        external_teacher_fail_open = True
    elif args.external_teacher_fail_open == "off":
        external_teacher_fail_open = False
    external_teacher_mode = None if args.external_teacher_mode == "default" else args.external_teacher_mode
    return {
        "text_hint": str(args.text_hint or ""),
        "tick_interval_ms": int(args.interval_ms or 0),
        "reset_runtime": bool(args.reset_runtime),
        "label": str(args.label or ""),
        "max_ticks": int(args.max_ticks or 0),
        "stop_on_capture_failures": int(args.stop_on_capture_failures or 0),
        "stop_on_action_errors": int(args.stop_on_action_errors or 0),
        "stop_on_idle_ticks": int(args.stop_on_idle_ticks or 0),
        "idle_backoff_ms": int(args.idle_backoff_ms or 0),
        "auto_feedback_enabled": auto_feedback_enabled,
        "teacher_mode": teacher_mode,
        "llm_gate_mode": llm_gate_mode,
        "external_teacher_enabled": external_teacher_enabled,
        "external_teacher_mode": external_teacher_mode,
        "external_teacher_stub_response_path": str(args.external_teacher_stub or ""),
        "external_teacher_fail_open": external_teacher_fail_open,
        "external_teacher_max_retries": (int(args.external_teacher_max_retries) if args.external_teacher_max_retries is not None and int(args.external_teacher_max_retries) > 0 else None),
        "external_teacher_retry_backoff_ms": (int(args.external_teacher_retry_backoff_ms) if args.external_teacher_retry_backoff_ms is not None else None),
        "external_teacher_http_endpoint": str(args.external_teacher_http_endpoint or ""),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "serve"
    config = load_config()
    app = ObservatoryV2App(config=config)

    if command == "serve":
        host = args.host or config.host
        port = int(args.port or config.port)
        run_server(app, host=host, port=port, open_browser=not bool(args.no_browser) and bool(config.auto_open_browser))
        return

    if command == "run-demo":
        result = app.start_demo_run(tick_count=args.ticks, tick_interval_ms=args.interval_ms, label=args.label)
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-text":
        texts = args.texts or ["今天 天气 不错", "今天 天气 不错", "我 想 出门"]
        result = app.start_text_run(texts=texts, label=args.label, tick_interval_ms=args.interval_ms, reset_runtime=bool(args.reset_runtime))
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-multimodal":
        items = _align_multimodal_inputs(args.texts or [], args.images or [], args.audios or [])
        result = app.start_multimodal_run(items=items, label=args.label, tick_interval_ms=args.interval_ms, reset_runtime=bool(args.reset_runtime))
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-dataset":
        payload = run_dataset_file(
            Path(args.dataset),
            default_label=str(args.label or "Phase10 命令行批量实验"),
            timeout_sec=float(args.timeout_sec),
            repo_root_value=app.repo_root,
            outputs_root_override=(str(args.outputs_root or "").strip() or None),
        )
        _print_json(payload)
        return

    if command == "run-screen":
        items = [{"text": args.text, "source_type": "screen_capture_run", "capture_screen": True} for _ in range(max(1, int(args.ticks)))]
        result = app.start_multimodal_run(
            items=items,
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
            run_kind="phase17_screen_capture_run",
        )
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-audio-stream":
        result = app.start_audio_stream_run(
            audio_bytes=Path(args.audio).read_bytes(),
            text_prefix=args.text_prefix,
            tick_window_ms=(args.tick_window_ms or None),
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
        )
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-image-stream":
        frame_bytes_list = [Path(path).read_bytes() for path in (args.frames or [])]
        strip_image_bytes = Path(args.strip_image).read_bytes() if str(args.strip_image or "").strip() else None
        result = app.start_image_stream_run(
            frame_bytes_list=frame_bytes_list or None,
            strip_image_bytes=strip_image_bytes,
            frame_count=args.frame_count,
            text_prefix=args.text_prefix,
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
        )
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-video-stream":
        result = app.start_video_stream_run(
            video_bytes=Path(args.video).read_bytes(),
            video_name=Path(args.video).name,
            text_prefix=args.text_prefix,
            tick_fps=(args.tick_fps or None),
            frame_stride=(args.frame_stride or None),
            max_frames=(args.max_frames or None),
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
        )
        app.wait_for_idle(timeout_sec=120.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-webcam-stream":
        result = app.start_webcam_stream_run(
            text_prefix=args.text_prefix,
            max_frames=(args.max_frames or None),
            device_index=int(args.device_index or 0),
            frame_width=(args.frame_width or None),
            frame_height=(args.frame_height or None),
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
        )
        app.wait_for_idle(timeout_sec=120.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-microphone-stream":
        device_index = None if int(args.device_index) < 0 else int(args.device_index)
        result = app.start_microphone_stream_run(
            text_prefix=args.text_prefix,
            max_windows=(args.max_windows or None),
            tick_window_ms=(args.tick_window_ms or None),
            sample_rate=int(args.sample_rate or 16000),
            channels=int(args.channels or 1),
            device_index=device_index,
            label=args.label,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
        )
        app.wait_for_idle(timeout_sec=120.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "run-autonomous":
        auto_feedback_enabled = None
        if args.auto_feedback == "on":
            auto_feedback_enabled = True
        elif args.auto_feedback == "off":
            auto_feedback_enabled = False
        teacher_mode = None if args.teacher_mode == "default" else args.teacher_mode
        llm_gate_mode = None if args.llm_gate_mode == "default" else args.llm_gate_mode
        external_teacher_enabled = None
        if args.external_teacher == "on":
            external_teacher_enabled = True
        elif args.external_teacher == "off":
            external_teacher_enabled = False
        external_teacher_fail_open = None
        if args.external_teacher_fail_open == "on":
            external_teacher_fail_open = True
        elif args.external_teacher_fail_open == "off":
            external_teacher_fail_open = False
        external_teacher_mode = None if args.external_teacher_mode == "default" else args.external_teacher_mode
        external_teacher_stub = str(args.external_teacher_stub or "").strip() or None
        external_teacher_http_endpoint = str(args.external_teacher_http_endpoint or "").strip() or None
        result = app.start_autonomous_run(
            ticks=args.ticks,
            text_hint=args.text_hint,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
            label=args.label,
            stop_on_capture_failures=(args.stop_on_capture_failures or None),
            stop_on_action_errors=(args.stop_on_action_errors or None),
            stop_on_idle_ticks=(args.stop_on_idle_ticks or None),
            idle_backoff_ms=(args.idle_backoff_ms or None),
            auto_feedback_enabled=auto_feedback_enabled,
            teacher_mode=teacher_mode,
            llm_gate_mode=llm_gate_mode,
            external_teacher_enabled=external_teacher_enabled,
            external_teacher_mode=external_teacher_mode,
            external_teacher_stub_response_path=external_teacher_stub,
            external_teacher_fail_open=external_teacher_fail_open,
            external_teacher_max_retries=(int(args.external_teacher_max_retries) if args.external_teacher_max_retries is not None and int(args.external_teacher_max_retries) > 0 else None),
            external_teacher_retry_backoff_ms=(int(args.external_teacher_retry_backoff_ms) if args.external_teacher_retry_backoff_ms is not None else None),
            external_teacher_http_endpoint=external_teacher_http_endpoint,
        )
        app.wait_for_idle(timeout_sec=120.0)
        manifest = app.get_manifest(result["run_id"])
        print(json.dumps({"result": result, "manifest": manifest}, ensure_ascii=False, indent=2))
        return

    if command == "run-autonomous-session":
        payload = _autonomous_session_payload_from_args(args)
        if str(args.server_url or "").strip():
            result = _json_request(str(args.server_url), "/api/autonomous-session/start", payload=payload)
            if bool(args.wait):
                wait_result = _wait_remote_autonomous_session(str(args.server_url), timeout_sec=float(args.timeout_sec or 120.0))
                _print_json({"result": result, "wait": wait_result})
            else:
                _print_json(result)
            return
        auto_feedback_enabled = payload["auto_feedback_enabled"] if isinstance(payload.get("auto_feedback_enabled"), bool) else None
        teacher_mode = str(payload.get("teacher_mode") or "") or None
        llm_gate_mode = str(payload.get("llm_gate_mode") or "") or None
        external_teacher_enabled = payload["external_teacher_enabled"] if isinstance(payload.get("external_teacher_enabled"), bool) else None
        external_teacher_fail_open = payload["external_teacher_fail_open"] if isinstance(payload.get("external_teacher_fail_open"), bool) else None
        external_teacher_mode = str(payload.get("external_teacher_mode") or "") or None
        external_teacher_stub = str(payload.get("external_teacher_stub_response_path") or "").strip() or None
        external_teacher_max_retries = payload["external_teacher_max_retries"] if isinstance(payload.get("external_teacher_max_retries"), int) else None
        external_teacher_retry_backoff_ms = payload["external_teacher_retry_backoff_ms"] if isinstance(payload.get("external_teacher_retry_backoff_ms"), int) else None
        external_teacher_http_endpoint = str(payload.get("external_teacher_http_endpoint") or "").strip() or None
        result = app.start_autonomous_session(
            text_hint=args.text_hint,
            tick_interval_ms=args.interval_ms,
            reset_runtime=bool(args.reset_runtime),
            label=args.label,
            max_ticks=(args.max_ticks or None),
            stop_on_capture_failures=(args.stop_on_capture_failures or None),
            stop_on_action_errors=(args.stop_on_action_errors or None),
            stop_on_idle_ticks=(args.stop_on_idle_ticks or None),
            idle_backoff_ms=(args.idle_backoff_ms or None),
            auto_feedback_enabled=auto_feedback_enabled,
            teacher_mode=teacher_mode,
            llm_gate_mode=llm_gate_mode,
            external_teacher_enabled=external_teacher_enabled,
            external_teacher_mode=external_teacher_mode,
            external_teacher_stub_response_path=external_teacher_stub,
            external_teacher_fail_open=external_teacher_fail_open,
            external_teacher_max_retries=external_teacher_max_retries,
            external_teacher_retry_backoff_ms=external_teacher_retry_backoff_ms,
            external_teacher_http_endpoint=external_teacher_http_endpoint,
        )
        if bool(args.wait):
            idle = app.wait_for_idle(timeout_sec=float(args.timeout_sec or 120.0))
            status = app.get_autonomous_session_status()
            manifest = app.get_manifest(result["run_id"])
            if not idle and bool(status.get("active", False)):
                stop_result = app.stop_autonomous_session()
                app.wait_for_idle(timeout_sec=10.0)
                status = app.get_autonomous_session_status()
                _print_json({"result": result, "wait": {"ok": False, "timeout": True, "stop_result": stop_result, "status": status}, "manifest": manifest})
                return
            _print_json({"result": result, "wait": {"ok": bool(idle), "status": status}, "manifest": manifest})
        else:
            _print_json(result)
        return

    if command == "pause-autonomous-session":
        if str(args.server_url or "").strip():
            _print_json(_json_request(str(args.server_url), "/api/autonomous-session/pause", payload={}))
            return
        _print_json(app.pause_autonomous_session())
        return

    if command == "resume-autonomous-session":
        if str(args.server_url or "").strip():
            _print_json(_json_request(str(args.server_url), "/api/autonomous-session/resume", payload={}))
            return
        _print_json(app.resume_autonomous_session())
        return

    if command == "stop-autonomous-session":
        if str(args.server_url or "").strip():
            _print_json(_json_request(str(args.server_url), "/api/autonomous-session/stop", payload={}))
            return
        _print_json(app.stop_autonomous_session())
        return

    if command == "recover-autonomous-session":
        payload = {"run_id": str(args.run_id or "")}
        if str(args.server_url or "").strip():
            _print_json(_json_request(str(args.server_url), "/api/autonomous-session/recover", payload=payload))
            return
        _print_json(app.recover_autonomous_session(run_id=str(args.run_id or "") or None))
        return

    if command == "autonomous-session-status":
        if str(args.server_url or "").strip():
            _print_json(_json_request(str(args.server_url), "/api/autonomous-session/status"))
            return
        _print_json(app.get_autonomous_session_status())
        return

    if command == "continue-from-checkpoint":
        result = app.continue_from_checkpoint(
            checkpoint_path=Path(args.infile),
            texts=args.texts,
            label=args.label,
            tick_interval_ms=args.interval_ms,
        )
        app.wait_for_idle(timeout_sec=60.0)
        manifest = app.get_manifest(result["run_id"])
        _print_json({"result": result, "manifest": manifest})
        return

    if command == "latest-run":
        run_id = app.latest_run_id()
        if not run_id:
            _print_json({})
            return
        _print_json(app.get_manifest(run_id))
        return

    if command == "export-runtime":
        payload = app.export_runtime()
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_json({"ok": True, "path": args.out})
        return

    if command == "import-runtime":
        payload = json.loads(Path(args.infile).read_text(encoding="utf-8"))
        _print_json(app.import_runtime(payload))
        return

    if command == "export-memory-bundle":
        _print_json(app.export_memory_deployment_bundle(Path(args.out_dir)))
        return

    if command == "inspect-memory-bundle":
        _print_json(app.inspect_memory_deployment_bundle(Path(args.directory)))
        return

    if command == "import-memory-bundle":
        _print_json(app.import_memory_deployment_bundle(Path(args.directory)))
        return

    if command == "forget":
        _print_json(
            app.forget_cold_memories(
                keep_latest=args.keep_latest,
                min_reality_weight=float(args.min_reality_weight or 0.0),
                min_total_item_energy=float(args.min_total_item_energy or 0.0),
                protect_memory_kinds=list(args.protect_memory_kinds or []),
                max_memory_count=(int(args.max_memory_count) if int(args.max_memory_count or 0) > 0 else None),
                strategy=str(args.strategy or "latest_only"),
                dry_run=bool(args.dry_run),
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
