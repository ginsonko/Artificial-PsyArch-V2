# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "dynamic_vision_coupling"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _frame(
    *,
    base_rect: tuple[int, int, int, int] | None = (68, 76, 108, 116),
    moving_rect: tuple[int, int, int, int] | None = None,
    size: tuple[int, int] = (192, 192),
) -> Image.Image:
    image = Image.new("RGB", size, color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    if base_rect is not None:
        draw.rectangle(base_rect, fill=(236, 236, 236))
    if moving_rect is not None:
        draw.rectangle(moving_rect, fill=(255, 255, 255))
    return image


def _build_runtime(*, dynamic_enabled: bool) -> RuntimeV2:
    overrides = {
        "autonomous_teacher_enabled": False,
        "autonomous_llm_gate_enabled": False,
        "autonomous_external_teacher_enabled": False,
        "intrinsic_feedback_enabled": True,
        "executor_enabled": False,
        "vision_attention_boost_enabled": True,
        "vision_attention_boost_decay": 0.72,
        "vision_patch_budget": 16,
        "vision_focus_patch_budget": 8,
        "vision_raw_state_budget": 64,
        "vision_reconstruction_patch_budget": 1024,
        "vision_attention_boost_max_extra_raw_budget": 192,
        "vision_attention_boost_max_extra_focus_budget": 8,
        "vision_attention_boost_min_radius_scale": 0.28,
        "vision_attention_boost_edge_gain": 1.35,
        "vision_attention_boost_gaze_sigma_scale": 0.52,
        "vision_dynamic_track_window": 6,
        "vision_dynamic_candidate_limit_background": 12,
        "vision_dynamic_candidate_limit_focus": 28,
        "vision_dynamic_track_limit": 40,
        "vision_dynamic_summary_limit": 4,
        "vision_dynamic_match_threshold": 0.46,
        "vision_dynamic_track_forget_ticks": 3,
    }
    runtime = RuntimeV2(config=load_config(overrides=overrides), repo_root=REPO_ROOT)
    if not dynamic_enabled:
        runtime.vision_sensor.dynamic_summary_limit = 0
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _run_tick(runtime: RuntimeV2, *, image: Image.Image, tick_index: int, source_type: str) -> dict[str, Any]:
    text_packet = runtime.text_sensor.ingest("", tick_index=tick_index, source_type=source_type)
    image_packet = runtime.vision_sensor.ingest_image_bytes(_png_bytes(image), tick_index=tick_index, source_type=source_type)
    started = time.perf_counter()
    tick = runtime.process_multimodal_tick(
        tick_index=tick_index,
        text_packet=text_packet,
        image_packet=image_packet,
        source_type=source_type,
    )
    selected_actions = list(((tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or []))
    runtime_action_effects = runtime.apply_selected_actions(selected_actions, runtime_tick=tick)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.set_last_logic_ms(elapsed_ms)
    tick["runtime_action_effects"] = runtime_action_effects
    tick["elapsed_ms"] = elapsed_ms
    return tick


def _summarize_tick(tick: dict[str, Any], *, tick_index: int, phase: str) -> dict[str, Any]:
    rules_result = dict(tick.get("rules_result", {}) or {})
    metrics = dict(rules_result.get("metrics_snapshot", {}) or {})
    raw_metrics = dict(rules_result.get("raw_metrics_snapshot", {}) or {})
    emotion = dict(rules_result.get("emotion_channels", {}) or {})
    raw_emotion = dict(rules_result.get("raw_emotion_channels", {}) or {})
    habituation = dict(rules_result.get("cognitive_feeling_habituation", {}) or {})
    habituation_gains = dict(habituation.get("gains", {}) or {})
    habituation_state = dict(habituation.get("state", {}) or {})
    image_packet = dict(tick.get("image_packet", {}) or {})
    dynamic_summary = dict(image_packet.get("dynamic_track_summary", {}) or {})
    attention_boost = dict((tick.get("runtime_action_effects", {}) or {}).get("attention_boost", {}) or {})
    auto_reorient = dict(rules_result.get("auto_visual_reorient", {}) or {})
    selected_actions = [str(item.get("action_name", "") or "") for item in (rules_result.get("planned_selected_actions_preview", []) or [])]
    dynamic_motion_samples = [dict(item) for item in (image_packet.get("dynamic_motion_samples", []) or []) if isinstance(item, dict)]
    top_dynamic = dynamic_motion_samples[0] if dynamic_motion_samples else {}
    top_dynamic_coords = dict(top_dynamic.get("coords", {}) or {})
    top_dynamic_attrs = dict(top_dynamic.get("attributes", {}) or {})
    gaze_after = dict((tick.get("runtime_action_effects", {}) or {}).get("gaze_center_after", {}) or {})
    gaze_distance = None
    if "x" in gaze_after and "y" in gaze_after and "cx" in top_dynamic_coords and "cy" in top_dynamic_coords:
        gaze_distance = _round4(
            abs(float(gaze_after.get("x", 0.5) or 0.5) - float(top_dynamic_coords.get("cx", 0.5) or 0.5))
            + abs(float(gaze_after.get("y", 0.5) or 0.5) - float(top_dynamic_coords.get("cy", 0.5) or 0.5))
        )
    return {
        "tick_index": int(tick_index),
        "phase": phase,
        "elapsed_ms": _round4(float(tick.get("elapsed_ms", 0.0) or 0.0)),
        "raw_surprise": _round4(float(raw_emotion.get("surprise", emotion.get("surprise", 0.0)) or 0.0)),
        "surprise": _round4(float(emotion.get("surprise", 0.0) or 0.0)),
        "raw_dissonance": _round4(float(raw_emotion.get("dissonance", emotion.get("dissonance", 0.0)) or 0.0)),
        "dissonance": _round4(float(emotion.get("dissonance", 0.0) or 0.0)),
        "correctness": _round4(float(emotion.get("correctness", 0.0) or 0.0)),
        "grasp": _round4(float(emotion.get("grasp", 0.0) or 0.0)),
        "raw_underprediction_mass": _round4(float(raw_metrics.get("state.prediction_underprediction_mass", metrics.get("state.prediction_underprediction_mass", 0.0)) or 0.0)),
        "underprediction_mass": _round4(float(metrics.get("state.prediction_underprediction_mass", 0.0) or 0.0)),
        "overprediction_mass": _round4(float(metrics.get("state.prediction_overprediction_mass", 0.0) or 0.0)),
        "alignment_score": _round4(float(metrics.get("state.prediction_alignment_score", 0.0) or 0.0)),
        "grasp_score": _round4(float(metrics.get("state.prediction_grasp_score", 0.0) or 0.0)),
        "surprise_gain": _round4(float(habituation_gains.get("surprise", 1.0) or 1.0)),
        "dissonance_gain": _round4(float(habituation_gains.get("dissonance", 1.0) or 1.0)),
        "surprise_habituation": _round4(float(habituation_state.get("surprise", 0.0) or 0.0)),
        "dissonance_habituation": _round4(float(habituation_state.get("dissonance", 0.0) or 0.0)),
        "habituation_same_signature": bool(habituation.get("same_signature", False)),
        "raw_budget": int(image_packet.get("raw_state_budget", 0) or 0),
        "raw_count": int(len(image_packet.get("raw_samples", []) or [])),
        "focus_count": int(len(image_packet.get("focus_priority_samples", []) or [])),
        "dynamic_track_count": int(dynamic_summary.get("track_count", 0) or 0),
        "dynamic_object_count": int(dynamic_summary.get("object_count", 0) or 0),
        "dynamic_salience_mean": _round4(float(dynamic_summary.get("dynamic_salience_mean", 0.0) or 0.0)),
        "global_motion_speed": _round4(float(dynamic_summary.get("global_motion_speed", 0.0) or 0.0)),
        "selected_actions": selected_actions,
        "auto_reorient": bool(auto_reorient),
        "auto_reorient_target": dict(auto_reorient.get("target", {}) or {}),
        "boost_active": bool(attention_boost.get("active", False)),
        "boost_source": str(attention_boost.get("source_action", "") or ""),
        "gaze_after": gaze_after,
        "top_dynamic_sa": str(top_dynamic.get("sa_label", "") or ""),
        "top_dynamic_objectness": _round4(float(top_dynamic_attrs.get("dynamic_objectness", 0.0) or 0.0)),
        "top_dynamic_motion_speed": _round4(float(top_dynamic_attrs.get("motion_speed", 0.0) or 0.0)),
        "gaze_to_top_dynamic_l1": gaze_distance,
    }


def _run_protocol(*, dynamic_enabled: bool) -> dict[str, Any]:
    runtime = _build_runtime(dynamic_enabled=dynamic_enabled)
    frames: list[tuple[str, Image.Image]] = []
    for _ in range(4):
        frames.append(("static", _frame()))
    for dx in [0, 8, 16, 24, 32, 40]:
        frames.append(("motion", _frame(moving_rect=(130 + dx, 30, 154 + dx, 54))))
    rows: list[dict[str, Any]] = []
    for tick_index, (phase, image) in enumerate(frames):
        tick = _run_tick(runtime, image=image, tick_index=tick_index, source_type=f"dynamic_coupling::{phase}")
        rows.append(_summarize_tick(tick, tick_index=tick_index, phase=phase))

    static_rows = [row for row in rows if row["phase"] == "static"]
    motion_rows = [row for row in rows if row["phase"] == "motion"]
    return {
        "dynamic_enabled": bool(dynamic_enabled),
        "rows": rows,
        "summary": {
            "static_mean_raw_surprise": _round4(sum(float(row["raw_surprise"]) for row in static_rows) / max(1, len(static_rows))),
            "static_mean_surprise": _round4(sum(float(row["surprise"]) for row in static_rows) / max(1, len(static_rows))),
            "motion_mean_raw_surprise": _round4(sum(float(row["raw_surprise"]) for row in motion_rows) / max(1, len(motion_rows))),
            "motion_mean_surprise": _round4(sum(float(row["surprise"]) for row in motion_rows) / max(1, len(motion_rows))),
            "static_mean_surprise_gain": _round4(sum(float(row["surprise_gain"]) for row in static_rows) / max(1, len(static_rows))),
            "motion_mean_surprise_gain": _round4(sum(float(row["surprise_gain"]) for row in motion_rows) / max(1, len(motion_rows))),
            "static_mean_raw_budget": _round4(sum(float(row["raw_budget"]) for row in static_rows) / max(1, len(static_rows))),
            "motion_mean_raw_budget": _round4(sum(float(row["raw_budget"]) for row in motion_rows) / max(1, len(motion_rows))),
            "static_auto_reorient_hits": int(sum(1 for row in static_rows if bool(row["auto_reorient"]))),
            "motion_auto_reorient_hits": int(sum(1 for row in motion_rows if bool(row["auto_reorient"]))),
            "motion_dynamic_object_peak": _round4(max((float(row["dynamic_object_count"]) for row in motion_rows), default=0.0)),
            "motion_dynamic_salience_peak": _round4(max((float(row["dynamic_salience_mean"]) for row in motion_rows), default=0.0)),
            "motion_best_gaze_to_dynamic": min(
                [float(row["gaze_to_top_dynamic_l1"]) for row in motion_rows if row.get("gaze_to_top_dynamic_l1") is not None] or [9.9999]
            ),
        },
    }


def _render_report(*, baseline: dict[str, Any], dynamic: dict[str, Any]) -> str:
    b = dict(baseline.get("summary", {}) or {})
    d = dict(dynamic.get("summary", {}) or {})
    lines = [
        "# V2 动态视觉联动实验结果",
        "",
        f"- 生成时间: {datetime.now().isoformat()}",
        "- 对照组: 关闭动态对象摘要参与（dynamic_summary_limit = 0）",
        "- 实验组: 开启动态对象摘要参与当前运行链",
        "",
        "## 关键汇总",
        "",
        f"- 对照组静态平均 raw surprise: {b.get('static_mean_raw_surprise', 0.0)}",
        f"- 对照组静态平均 effective surprise: {b.get('static_mean_surprise', 0.0)}",
        f"- 对照组静态平均 surprise gain: {b.get('static_mean_surprise_gain', 0.0)}",
        f"- 对照组运动平均 raw surprise: {b.get('motion_mean_raw_surprise', 0.0)}",
        f"- 对照组运动平均 effective surprise: {b.get('motion_mean_surprise', 0.0)}",
        f"- 对照组运动平均 surprise gain: {b.get('motion_mean_surprise_gain', 0.0)}",
        f"- 对照组运动期自动回看次数: {b.get('motion_auto_reorient_hits', 0)}",
        f"- 对照组运动期平均 raw budget: {b.get('motion_mean_raw_budget', 0.0)}",
        f"- 对照组运动期最佳 gaze->dynamic 距离: {b.get('motion_best_gaze_to_dynamic', 0.0)}",
        "",
        f"- 实验组静态平均 raw surprise: {d.get('static_mean_raw_surprise', 0.0)}",
        f"- 实验组静态平均 effective surprise: {d.get('static_mean_surprise', 0.0)}",
        f"- 实验组静态平均 surprise gain: {d.get('static_mean_surprise_gain', 0.0)}",
        f"- 实验组运动平均 raw surprise: {d.get('motion_mean_raw_surprise', 0.0)}",
        f"- 实验组运动平均 effective surprise: {d.get('motion_mean_surprise', 0.0)}",
        f"- 实验组运动平均 surprise gain: {d.get('motion_mean_surprise_gain', 0.0)}",
        f"- 实验组运动期自动回看次数: {d.get('motion_auto_reorient_hits', 0)}",
        f"- 实验组运动期平均 raw budget: {d.get('motion_mean_raw_budget', 0.0)}",
        f"- 实验组运动期动态对象峰值: {d.get('motion_dynamic_object_peak', 0.0)}",
        f"- 实验组运动期动态显著度峰值: {d.get('motion_dynamic_salience_peak', 0.0)}",
        f"- 实验组运动期最佳 gaze->dynamic 距离: {d.get('motion_best_gaze_to_dynamic', 0.0)}",
        "",
        "## 初步解释",
        "",
        "1. 这轮实验主要看三件事：",
        "   - 运动出现后，是否持续产出动态对象摘要；",
        "   - 惊/违和是否会触发自动视觉回看；",
        "   - 自动回看后，下一 tick 的采样预算是否被拉高。",
        "",
        "2. 如果实验组的 gaze->dynamic 距离明显更小，说明“运动对象驱动注意力”这一点已经开始成立。",
        "",
        "3. 如果 raw surprise 仍高、但 effective surprise 已经回落，说明底层新异检测还在工作，但系统开始对持续重复刺激形成习惯化，不再把它们全部当成同等强度的惊。",
        "",
        "## 备注",
        "",
        "- 这轮实验证明的是联动基础链，不是最终 OCR-like 连续视觉识别成功率。",
        "- 这轮实验结果应和逐 tick 明细 JSON 一起阅读。",
    ]
    return "\n".join(lines)


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_protocol(dynamic_enabled=False)
    dynamic = _run_protocol(dynamic_enabled=True)
    summary = {
        "schema_id": "dynamic_vision_coupling/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "baseline": baseline,
        "dynamic": dynamic,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_text(output_dir / "report.md", _render_report(baseline=baseline, dynamic=dynamic))
    print(output_dir)


if __name__ == "__main__":
    main()
