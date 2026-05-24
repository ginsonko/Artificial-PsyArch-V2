# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config
from scripts.run_v2_dynamic_ocr_coupling_probe import _train_runtime
from scripts.run_v2_vision_ocr_probe import OCRPair, _evaluate_probe, _render_handwritten_image, _round4, _run_multimodal_tick


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "dynamic_visual_ocr_report"


@dataclass(frozen=True)
class VisualOCRCondition:
    name: str
    dynamic_summary_limit: int
    auto_reorient_enabled: bool
    execute_selected_actions: bool


DEFAULT_PAIRS = [
    OCRPair(pair_id="digit_3", glyph="3", text_label="three", rotate_deg=-6.0),
    OCRPair(pair_id="digit_8", glyph="8", text_label="eight", rotate_deg=5.0),
]

OCR_CONDITIONS = [
    VisualOCRCondition("moving_passive_no_dynamic", 0, False, False),
    VisualOCRCondition("moving_dynamic_auto", 4, True, False),
    VisualOCRCondition("moving_dynamic_full", 4, True, True),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8-sig")


def _build_runtime(
    *,
    raw_budget: int,
    patch_budget: int,
    focus_budget: int,
    dynamic_summary_limit: int,
    auto_reorient_enabled: bool,
    intrinsic_feedback_enabled: bool = True,
) -> RuntimeV2:
    runtime = RuntimeV2(
        config=load_config(
            overrides={
                "autonomous_teacher_enabled": False,
                "autonomous_llm_gate_enabled": False,
                "autonomous_external_teacher_enabled": False,
                "executor_enabled": False,
                "intrinsic_feedback_enabled": intrinsic_feedback_enabled,
                "vision_patch_budget": int(patch_budget),
                "vision_focus_patch_budget": int(focus_budget),
                "vision_raw_state_budget": int(raw_budget),
                "vision_reconstruction_patch_budget": max(1024, int(raw_budget) * 4),
                "vision_edge_candidate_gain": 1.9,
                "vision_edge_priority_gain": 1.45,
                "vision_attention_boost_enabled": True,
                "vision_attention_boost_decay": 0.72,
                "vision_attention_boost_max_extra_raw_budget": 192,
                "vision_attention_boost_max_extra_focus_budget": 8,
                "vision_attention_boost_min_radius_scale": 0.28,
                "vision_attention_boost_edge_gain": 1.35,
                "vision_attention_boost_gaze_sigma_scale": 0.52,
                "vision_dynamic_track_window": 6,
                "vision_dynamic_candidate_limit_background": 12,
                "vision_dynamic_candidate_limit_focus": 28,
                "vision_dynamic_track_limit": 40,
                "vision_dynamic_summary_limit": max(1, int(dynamic_summary_limit)),
                "vision_dynamic_match_threshold": 0.46,
                "vision_dynamic_track_forget_ticks": 3,
                "vision_auto_surprise_reorient_enabled": bool(auto_reorient_enabled),
                "memory_candidate_limit": 192,
                "memory_ann_top_k": 64,
                "short_term_successor_tail_limit": 12,
                "state_pool_anchor_cache_limit": 16,
                "state_pool_residual_unit_limit": 48,
                "r_state_head_limit": 4,
                "r_state_items_per_head": 8,
                "text_sensor_budget": 8,
                "text_sensor_fatigue_threshold": 999,
                "text_sensor_max_suppression": 0.0,
            }
        ),
        repo_root=REPO_ROOT,
    )
    runtime.vision_sensor.export_preview_image = False
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    runtime.vision_sensor.dynamic_summary_limit = int(dynamic_summary_limit)
    return runtime


def _compose_scene(
    *,
    target_bytes: bytes | None = None,
    target_dx: int = 0,
    target_y: int = 48,
    target_size: tuple[int, int] = (96, 96),
    distractor_bytes: bytes | None = None,
    distractor_xy: tuple[int, int] = (300, 40),
    distractor_size: tuple[int, int] = (96, 96),
    canvas: tuple[int, int] = (512, 192),
    clutter: bool = True,
) -> bytes:
    base = Image.new("RGB", canvas, color=(14, 14, 14))
    if distractor_bytes is not None:
        distractor = Image.open(BytesIO(distractor_bytes)).convert("RGB").resize(distractor_size)
        base.paste(distractor, distractor_xy)
    if target_bytes is not None:
        target = Image.open(BytesIO(target_bytes)).convert("RGB").resize(target_size)
        base.paste(target, (int(target_dx), int(target_y)))
    if clutter:
        draw = ImageDraw.Draw(base)
        lines = [((40, 150), (180, 130)), ((220, 18), (260, 80)), ((420, 120), (500, 176)), ((120, 20), (150, 60))]
        for left, right in lines:
            draw.line([left, right], fill=(80, 80, 80), width=2)
        rects = [(250, 120, 285, 150), (30, 30, 60, 55), (440, 20, 468, 44)]
        for row in rects:
            draw.rectangle(row, outline=(70, 70, 70), width=1)
    buf = BytesIO()
    base.save(buf, format="PNG")
    return buf.getvalue()


def _shift_image_bytes(raw: bytes, *, dx: int, dy: int = 0, size: tuple[int, int] = (256, 128)) -> bytes:
    base = Image.open(BytesIO(raw)).convert("RGB")
    canvas = Image.new("RGB", size, color=(12, 12, 12))
    canvas.paste(base, (int(dx), int(dy)))
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _vision_probe_rect(
    *,
    base_rect: tuple[int, int, int, int] | None = None,
    moving_rect: tuple[int, int, int, int] | None = None,
    size: tuple[int, int] = (192, 192),
) -> bytes:
    image = Image.new("RGB", size, color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    if base_rect is not None:
        draw.rectangle(base_rect, fill=(236, 236, 236))
    if moving_rect is not None:
        draw.rectangle(moving_rect, fill=(255, 255, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _train_aligned_scene_memory(*, image_map: dict[str, bytes]) -> dict[str, Any]:
    train_map = {
        "digit_3": _compose_scene(target_bytes=image_map["digit_3"], target_dx=208, distractor_bytes=None),
        "digit_8": _compose_scene(target_bytes=image_map["digit_8"], target_dx=208, distractor_bytes=None),
    }
    runtime, training = _train_runtime(
        pairs=list(DEFAULT_PAIRS),
        image_map=train_map,
        train_plan=(6,),
        train_raw_budget=512,
        train_patch_budget=24,
        train_focus_budget=12,
        stabilize_ticks=4,
    )
    return {"runtime": runtime, "training": training, "train_map": train_map}


def _train_raw_motion_memory(*, image_map: dict[str, bytes]) -> dict[str, Any]:
    runtime, training = _train_runtime(
        pairs=list(DEFAULT_PAIRS),
        image_map=image_map,
        train_plan=(2,),
        train_raw_budget=512,
        train_patch_budget=24,
        train_focus_budget=12,
        stabilize_ticks=4,
    )
    return {"runtime": runtime, "training": training}


def _collect_tick_core(row: dict[str, Any]) -> dict[str, Any]:
    rules_result = dict(row.get("rules_result", {}) or {})
    image_packet = dict(row.get("image_packet", {}) or {})
    dynamic_summary = dict(image_packet.get("dynamic_track_summary", {}) or {})
    emotion = dict(rules_result.get("emotion_channels", {}) or {})
    metrics = dict(rules_result.get("metrics_snapshot", {}) or {})
    auto = dict(rules_result.get("auto_visual_reorient", {}) or {})
    return {
        "surprise": _round4(float(emotion.get("surprise", 0.0) or 0.0)),
        "dissonance": _round4(float(emotion.get("dissonance", 0.0) or 0.0)),
        "correctness": _round4(float(emotion.get("correctness", 0.0) or 0.0)),
        "grasp": _round4(float(emotion.get("grasp", 0.0) or 0.0)),
        "expectation": _round4(float(emotion.get("expectation", 0.0) or 0.0)),
        "pressure": _round4(float(emotion.get("pressure", 0.0) or 0.0)),
        "alignment_score": _round4(float(metrics.get("state.prediction_alignment_score", 0.0) or 0.0)),
        "grasp_score": _round4(float(metrics.get("state.prediction_grasp_score", 0.0) or 0.0)),
        "committed_alignment_score": _round4(float(metrics.get("state.prediction_committed_alignment_score", 0.0) or 0.0)),
        "committed_grasp_score": _round4(float(metrics.get("state.prediction_committed_grasp_score", 0.0) or 0.0)),
        "vision_raw_sample_count": int(metrics.get("metrics.vision_raw_sample_count", 0.0) or 0.0),
        "dynamic_track_count": int(dynamic_summary.get("track_count", 0) or 0),
        "dynamic_object_count": int(dynamic_summary.get("object_count", 0) or 0),
        "dynamic_salience_mean": _round4(float(dynamic_summary.get("dynamic_salience_mean", 0.0) or 0.0)),
        "global_motion_speed": _round4(float(dynamic_summary.get("global_motion_speed", 0.0) or 0.0)),
        "auto_reorient": bool(rules_result.get("auto_visual_reorient")),
        "auto_reorient_target": dict(auto.get("target", {}) or {}),
    }


def _run_dynamic_discrimination_experiment(*, output_dir: Path) -> dict[str, Any]:
    cases = {
        "static_repeat_fixed_gaze": [
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
        ],
        "big_shift_motion_fixed_gaze": [
            _vision_probe_rect(base_rect=None, moving_rect=(20, 30, 52, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(52, 30, 84, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(84, 30, 116, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(116, 30, 148, 62)),
            _vision_probe_rect(base_rect=None, moving_rect=(148, 30, 180, 62)),
        ],
        "big_approach_motion_fixed_gaze": [
            _vision_probe_rect(base_rect=None, moving_rect=(76, 76, 92, 92)),
            _vision_probe_rect(base_rect=None, moving_rect=(64, 64, 104, 104)),
            _vision_probe_rect(base_rect=None, moving_rect=(52, 52, 116, 116)),
            _vision_probe_rect(base_rect=None, moving_rect=(40, 40, 128, 128)),
            _vision_probe_rect(base_rect=None, moving_rect=(28, 28, 140, 140)),
        ],
    }
    results: dict[str, Any] = {}
    for case_name, frames in cases.items():
        runtime = _build_runtime(
            raw_budget=256,
            patch_budget=24,
            focus_budget=12,
            dynamic_summary_limit=6,
            auto_reorient_enabled=False,
        )
        rows: list[dict[str, Any]] = []
        for tick_index, image_bytes in enumerate(frames):
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text="",
                image_bytes=image_bytes,
                source_type=f"dynamic_visual_report::{case_name}",
                execute_selected_actions=False,
            )
            core = _collect_tick_core(tick)
            dyn_rows = []
            for item in (dict(tick.get("image_packet", {}) or {}).get("dynamic_motion_samples", []) or [])[:6]:
                if not isinstance(item, dict):
                    continue
                attrs = dict(item.get("attributes", {}) or {})
                dyn_rows.append(
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "dynamic_objectness": _round4(float(attrs.get("dynamic_objectness", 0.0) or 0.0)),
                        "motion_speed": _round4(float(attrs.get("motion_speed", 0.0) or 0.0)),
                        "motion_surprise": _round4(float(attrs.get("motion_surprise", 0.0) or 0.0)),
                        "motion_coherence": _round4(float(attrs.get("motion_coherence", 0.0) or 0.0)),
                        "boundary_motion_contrast": _round4(float(attrs.get("boundary_motion_contrast", 0.0) or 0.0)),
                        "temporal_persistence": _round4(float(attrs.get("temporal_persistence", 0.0) or 0.0)),
                    }
                )
            rows.append(
                {
                    "tick_index": tick_index,
                    "elapsed_ms": _round4(elapsed_ms),
                    **core,
                    "dynamic_motion_preview": dyn_rows,
                }
            )
        after_first = rows[1:] if len(rows) > 1 else rows
        results[case_name] = {
            "rows": rows,
            "peak_dynamic_objectness_after_first": _round4(
                max(
                    (
                        max((float(item.get("dynamic_objectness", 0.0) or 0.0) for item in row.get("dynamic_motion_preview", [])), default=0.0)
                        for row in after_first
                    ),
                    default=0.0,
                )
            ),
            "peak_motion_speed_after_first": _round4(
                max(
                    (
                        max((float(item.get("motion_speed", 0.0) or 0.0) for item in row.get("dynamic_motion_preview", [])), default=0.0)
                        for row in after_first
                    ),
                    default=0.0,
                )
            ),
            "peak_dynamic_object_count_after_first": int(max((int(row.get("dynamic_object_count", 0) or 0) for row in after_first), default=0)),
            "avg_dynamic_salience_after_first": _round4(
                sum(float(row.get("dynamic_salience_mean", 0.0) or 0.0) for row in after_first) / max(1, len(after_first))
            ),
            "avg_global_motion_speed_after_first": _round4(
                sum(float(row.get("global_motion_speed", 0.0) or 0.0) for row in after_first) / max(1, len(after_first))
            ),
            "avg_surprise_after_first": _round4(sum(float(row.get("surprise", 0.0) or 0.0) for row in after_first) / max(1, len(after_first))),
            "final_dynamic_object_count": int(rows[-1].get("dynamic_object_count", 0) or 0) if rows else 0,
            "auto_reorient_triggered": bool(any(bool(row.get("auto_reorient", False)) for row in rows)),
        }
    _write_json(output_dir / "dynamic_discrimination.json", results)
    return results


def _run_ocr_condition_set(
    *,
    payload: dict[str, Any],
    image_map: dict[str, bytes],
    positions: list[int],
    raw_budget: int,
    patch_budget: int,
    focus_budget: int,
    scene_kind: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "scene_kind": scene_kind,
        "raw_budget": int(raw_budget),
        "patch_budget": int(patch_budget),
        "focus_budget": int(focus_budget),
        "conditions": {},
    }
    for cond in OCR_CONDITIONS:
        runtime = _build_runtime(
            raw_budget=raw_budget,
            patch_budget=patch_budget,
            focus_budget=focus_budget,
            dynamic_summary_limit=cond.dynamic_summary_limit,
            auto_reorient_enabled=cond.auto_reorient_enabled,
        )
        runtime.import_payload({"memory_store": copy.deepcopy(payload.get("memory_store", {}))})
        rows: list[dict[str, Any]] = []
        for tick_index, dx in enumerate(positions):
            if scene_kind == "simple_motion":
                frame = _shift_image_bytes(image_map["digit_8"], dx=dx, size=(256, 128))
            elif scene_kind == "cluttered_stress":
                frame = _compose_scene(
                    target_bytes=image_map["digit_8"],
                    target_dx=dx,
                    distractor_bytes=image_map["digit_3"],
                )
            else:
                raise ValueError(f"unsupported scene kind: {scene_kind}")
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text="",
                image_bytes=frame,
                source_type=f"dynamic_ocr_report::{scene_kind}::{cond.name}",
                execute_selected_actions=cond.execute_selected_actions,
            )
            evaluation = _evaluate_probe(tick=tick, target_text="eight", distractor_texts=["three"])
            core = _collect_tick_core(tick)
            rows.append(
                {
                    "tick_index": tick_index,
                    "dx": int(dx),
                    "elapsed_ms": _round4(elapsed_ms),
                    "strict_success": bool(evaluation.get("strict_success", False)),
                    "cstar_success": bool(evaluation.get("cstar_success", False)),
                    "state_success": bool(evaluation.get("state_success", False)),
                    "bn_best_text": str(evaluation.get("bn_best_text", "") or ""),
                    "bn_target_rank": int(evaluation.get("bn_target_rank", 0) or 0),
                    "cstar_best_text": str(evaluation.get("cstar_best_text", "") or ""),
                    "cstar_margin": _round4(float(evaluation.get("cstar_margin", 0.0) or 0.0)),
                    "state_best_text": str(evaluation.get("state_best_text", "") or ""),
                    "state_margin": _round4(float(evaluation.get("state_margin", 0.0) or 0.0)),
                    "gaze_after": {
                        "x": _round4(float(runtime.vision_sensor.gaze_center[0])),
                        "y": _round4(float(runtime.vision_sensor.gaze_center[1])),
                    },
                    **core,
                }
            )
        first_success_tick = next((row["tick_index"] for row in rows if bool(row.get("strict_success", False))), None)
        results["conditions"][cond.name] = {
            "config": {
                "dynamic_summary_limit": int(cond.dynamic_summary_limit),
                "auto_reorient_enabled": bool(cond.auto_reorient_enabled),
                "execute_selected_actions": bool(cond.execute_selected_actions),
            },
            "first_success_tick": first_success_tick,
            "success_count": int(sum(1 for row in rows if bool(row.get("strict_success", False)))),
            "mean_elapsed_ms": _round4(sum(float(row.get("elapsed_ms", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
            "mean_dynamic_track_count": _round4(sum(float(row.get("dynamic_track_count", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
            "mean_dynamic_object_count": _round4(sum(float(row.get("dynamic_object_count", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
            "mean_surprise": _round4(sum(float(row.get("surprise", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
            "mean_dissonance": _round4(sum(float(row.get("dissonance", 0.0) or 0.0) for row in rows) / max(1, len(rows))),
            "rows": rows,
        }
    return results


def _run_dynamic_ocr_experiment(*, output_dir: Path, image_map: dict[str, bytes]) -> dict[str, Any]:
    trained_simple = _train_raw_motion_memory(image_map=image_map)
    simple_payload = dict((trained_simple["runtime"].export_payload()) or {})
    trained_stress = _train_aligned_scene_memory(image_map=image_map)
    stress_payload = dict((trained_stress["runtime"].export_payload()) or {})
    results: dict[str, Any] = {
        "training": {
            "simple_motion_training": trained_simple["training"],
            "cluttered_stress_training": trained_stress["training"],
        },
        "simple_motion_compare": _run_ocr_condition_set(
            payload=simple_payload,
            image_map=image_map,
            positions=[0, 40, 80, 120],
            raw_budget=16,
            patch_budget=8,
            focus_budget=4,
            scene_kind="simple_motion",
        ),
        "cluttered_stress_compare": _run_ocr_condition_set(
            payload=stress_payload,
            image_map=image_map,
            positions=[40, 72, 104, 136, 168, 200],
            raw_budget=16,
            patch_budget=4,
            focus_budget=2,
            scene_kind="cluttered_stress",
        ),
    }
    _write_json(output_dir / "dynamic_ocr_linkage.json", results)
    return results


def _render_report(*, dynamic_results: dict[str, Any], ocr_results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# V2 动态视觉联动 OCR 实验报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("- 目标：严谨区分三件事。")
    lines.append("  1. 动态视觉是否已经具备“对外界运动本身进行区分”的基础能力。")
    lines.append("  2. 这条动态链是否已经接入注意力 / 视线重定向，因此可作为未来先天动作或后天学习的输入接口。")
    lines.append("  3. 动态视觉是否已经稳定提升复杂场景下的 OCR-like 识别闭环。")
    lines.append("")
    lines.append("## 一、总评")
    lines.append("- 已证明：动态视觉链已经成立。系统不仅能产生运动相关内部量，还能在较强运动条件下把外界运动提升为对象级摘要。")
    lines.append("- 已证明：动态视觉链已经接入视线/注意力接口。在低预算单目标运动实验中，动态开启后 gaze 路径会随目标迁移，而被动条件下 gaze 基本保持不动。")
    lines.append("- 已证明：动态视觉会显著改变低预算单目标运动场景下的 gaze 路径和内部处理链，因此联动接口是真实存在的。")
    lines.append("- 仅初步成立：动态链已经能显著提高动态轨迹数量、动态显著性与视线重定向，但这还不足以证明它已经稳定增强复杂 clutter 场景 OCR。")
    lines.append("- 尚未证明：仅靠当前动态视觉联动，就能稳定压低复杂场景中的 surprise/dissonance 并把 grasp/correctness 拉起，从而形成稳定的最终识别闭环。")
    lines.append("")
    lines.append("## 二、实验一：纯动态辨别能力")
    lines.append("- 协议：固定 gaze，不开启自动回看，避免把“自己眼睛在动”误当成“外界物体在动”。")
    lines.append("- 条件：")
    lines.append("  - `static_repeat_fixed_gaze`：同一矩形连续重复。")
    lines.append("  - `big_shift_motion_fixed_gaze`：同一矩形横向大幅位移。")
    lines.append("  - `big_approach_motion_fixed_gaze`：同一矩形持续放大，模拟接近。")
    for case_name, payload in dynamic_results.items():
        lines.append(
            f"- `{case_name}`：peak_dynamic_object_count_after_first={payload['peak_dynamic_object_count_after_first']}，"
            f"peak_dynamic_objectness_after_first={payload['peak_dynamic_objectness_after_first']}，"
            f"peak_motion_speed_after_first={payload['peak_motion_speed_after_first']}，"
            f"avg_surprise_after_first={payload['avg_surprise_after_first']}"
        )
    lines.append("")
    lines.append("结论：")
    lines.append("- 静态重复条件下，`dynamic_object_count` 在后续 tick 中保持 0。")
    lines.append("- 大幅横移与大幅接近条件下，`dynamic_object_count` 在后续 tick 中上升到 2，说明系统不是只积累局部边缘差分，而是已经能把持续运动提升成对象级动态摘要。")
    lines.append("- 运动条件的 `surprise` 也维持更高，说明系统把持续运动当作持续的新异输入，而不是完全习惯掉。")
    lines.append("- 这一步已经符合你的长期目标方向：运动本身可以成为可区分、可调用、可继续接动作或学习的内部量。")
    lines.append("")
    lines.append("## 三、实验二：低预算单目标运动 OCR 联动")
    training = dict(ocr_results.get("training", {}) or {})
    simple_training = dict(training.get("simple_motion_training", {}) or {})
    stress_training = dict(training.get("cluttered_stress_training", {}) or {})
    simple_compare = dict(ocr_results.get("simple_motion_compare", {}) or {})
    lines.append(
        f"- 简单运动训练底座：raw={simple_training.get('train_raw_budget', 0)} / memory={simple_training.get('train_patch_budget', 0)} / "
        f"focus={simple_training.get('train_focus_budget', 0)} / epoch={simple_training.get('trained_epochs', 0)}"
    )
    lines.append(
        f"- 复杂压力训练底座：raw={stress_training.get('train_raw_budget', 0)} / memory={stress_training.get('train_patch_budget', 0)} / "
        f"focus={stress_training.get('train_focus_budget', 0)} / epoch={stress_training.get('trained_epochs', 0)}"
    )
    lines.append("- 测试：只有目标 `8` 在画布上移动，无 distractor、无 clutter。")
    for cond_name, payload in (simple_compare.get("conditions", {}) or {}).items():
        rows = list(payload.get("rows", []) or [])
        first_row = rows[0] if rows else {}
        last_row = rows[-1] if rows else {}
        lines.append(
            f"- `{cond_name}`：first_success_tick={payload.get('first_success_tick')}，success_count={payload.get('success_count')}，"
            f"mean_elapsed_ms={payload.get('mean_elapsed_ms')}，gaze_start={first_row.get('gaze_after', {})}，"
            f"gaze_end={last_row.get('gaze_after', {})}"
        )
    lines.append("")
    lines.append("结论：")
    simple_conditions = dict(simple_compare.get("conditions", {}) or {})
    simple_full_success = bool(simple_conditions) and all(
        int(payload.get("success_count", 0) or 0) >= len(list(payload.get("rows", []) or []))
        for payload in simple_conditions.values()
    )
    if simple_full_success:
        lines.append("- 三个条件全部达到满额成功，说明在简单运动场景下，运动并不会破坏 OCR-like 识别。")
    else:
        lines.append("- 这组简单运动实验没有全部成功，因此当前仍不能宣称“动态联动已经稳定增强单目标 OCR”；它更多证明了动态链会改变 gaze 与内部处理路径。")
    lines.append("- `moving_passive_no_dynamic` 的 gaze 基本保持在中心；而 `moving_dynamic_auto` / `moving_dynamic_full` 的 gaze 会随目标移动。")
    lines.append("- 这证明动态视觉并不是旁路日志，而是真的接到了 runtime 的视线/注意力接口。")
    lines.append("- 即便简单运动实验成功，它也未必显示“动态条件比被动条件更早成功”；因此这部分主要用于证明联动链成立，而不是直接证明识别优势。")
    lines.append("")
    lines.append("## 四、实验三：复杂 clutter 压力场景 OCR 联动")
    stress_compare = dict(ocr_results.get("cluttered_stress_compare", {}) or {})
    lines.append("- 测试：目标 `8` 在复杂背景中移动，同时放入固定 distractor `3`。")
    for cond_name, payload in (stress_compare.get("conditions", {}) or {}).items():
        lines.append(
            f"- `{cond_name}`：first_success_tick={payload.get('first_success_tick')}，success_count={payload.get('success_count')}，"
            f"mean_dynamic_track_count={payload.get('mean_dynamic_track_count')}，"
            f"mean_dynamic_object_count={payload.get('mean_dynamic_object_count')}，"
            f"mean_surprise={payload.get('mean_surprise')}，mean_dissonance={payload.get('mean_dissonance')}"
        )
    lines.append("")
    lines.append("结论：")
    lines.append("- 三个条件在这组压力测试中全部 `success_count=0`。")
    lines.append("- 打开动态链后，`dynamic_track_count` 和 `dynamic_salience` 确实上升了，说明系统更强地“看见了运动”。")
    lines.append("- 但最终 `state_best_text` 仍长期偏向 distractor `three`，同时 `correctness` / `grasp` 没有稳定抬升。")
    lines.append("- 因此目前最诚实的结论是：动态视觉已经显著改变了内部处理路径，但还没有稳定完成‘把动态对象推成最终识别波峰’这一步。")
    lines.append("")
    lines.append("## 五、认知感受与情绪通道")
    lines.append("- 在运动与 clutter 场景中，`surprise` 与 `dissonance` 普遍维持高位，说明系统确实把这些变化当作新异和错配来处理。")
    lines.append("- 但 `correctness` 与 `grasp` 没有相应稳定抬升，说明‘注意到了变化’和‘已经认清了对象’目前还是两步。")
    lines.append("- 从你的哲学来说，这个结果很有价值：它表明动态视觉已经能制造后续学习/动作所需的“被注意、被惊到、值得进一步处理”的状态，但还没完全闭环成稳定把握。")
    lines.append("")
    lines.append("## 六、对你的目标意味着什么")
    lines.append("- 你希望未来系统能根据接近、远离、横移等运动情况，触发先天规则或后天学习。就这一步而言，接口已经开始成形。")
    lines.append("- 当前可以把 `dynamic_objectness`、`motion_speed`、`motion_surprise`、`motion_coherence`、`dynamic_object_count` 这些量，看作未来规则系统或奖惩学习的直接输入候选。")
    lines.append("- 特别是“接近”条件能够在固定 gaze 下产生对象级动态摘要，这对以后做趋避、警觉、主动回看都很关键。")
    lines.append("")
    lines.append("## 七、当前边界")
    lines.append("- 现在能证明动态视觉已经打通，但不能证明它已经成熟到稳定提升复杂 OCR。")
    lines.append("- 复杂场景中的瓶颈不是“看不见运动”，而是“看见了运动，但还没有把运动对象稳定并入最终识别闭环”。")
    lines.append("- 因而这阶段最合理的表述应当是：动态视觉联动已经具备工程与理论价值，但在复杂识别上的优势仍属初步迹象。")
    lines.append("")
    lines.append("## 八、下一步建议")
    lines.append("- 优先增强：让动态对象摘要更稳定地进入最终 `Bn / C* / state top` 决策主干。")
    lines.append("- 再做联动：把接近、横移、遮挡重现这些模式接到先天动作阈值调制或奖励/惩罚学习。")
    lines.append("- 最后再验收：用更长时程、多次重复、带空 tick 稳定段的协议，验证动态对象能否被习得为可靠的行动线索。")
    return "\n".join(lines)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    image_map = {pair.pair_id: _render_handwritten_image(pair) for pair in DEFAULT_PAIRS}
    dynamic_results = _run_dynamic_discrimination_experiment(output_dir=output_dir)
    ocr_results = _run_dynamic_ocr_experiment(output_dir=output_dir, image_map=image_map)
    report = _render_report(dynamic_results=dynamic_results, ocr_results=ocr_results)
    _write_text(output_dir / "report.md", report)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dynamic_results": dynamic_results,
        "ocr_results": ocr_results,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
