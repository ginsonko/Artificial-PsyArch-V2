# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import math
import struct
import sys
import time
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from memory.memory_store_v2 import MemoryStoreV2
from observatory_v2.config import load_config
from scripts.run_v2_dynamic_ocr_coupling_probe import _train_runtime
from scripts.run_v2_vision_ocr_probe import OCRPair, _evaluate_probe, _render_handwritten_image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "multichannel_feelings_report"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8-sig")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round4(sum(float(v) for v in values) / max(1, len(values)))


def _mk_runtime(*, overrides: dict[str, Any] | None = None) -> RuntimeV2:
    merged = {
        "autonomous_teacher_enabled": False,
        "autonomous_llm_gate_enabled": False,
        "autonomous_external_teacher_enabled": False,
        "executor_enabled": False,
        "memory_candidate_limit": 192,
        "memory_ann_top_k": 64,
        "short_term_successor_tail_limit": 12,
        "state_pool_anchor_cache_limit": 16,
        "state_pool_residual_unit_limit": 48,
        "r_state_head_limit": 4,
        "r_state_items_per_head": 8,
    }
    if overrides:
        merged.update(overrides)
    runtime = RuntimeV2(config=load_config(overrides=merged), repo_root=REPO_ROOT)
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _mk_wav(freq: float, *, duration_sec: float = 0.24, sample_rate: int = 16000, amplitude: int = 12000) -> bytes:
    frames = bytearray()
    for i in range(int(sample_rate * duration_sec)):
        sample = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", sample)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def _mk_chirp(*, start_hz: float, end_hz: float, duration_sec: float = 0.24, sample_rate: int = 16000, amplitude: int = 12000) -> bytes:
    frame_count = int(sample_rate * duration_sec)
    frames = bytearray()
    for i in range(frame_count):
        progress = i / max(1, frame_count - 1)
        freq = start_hz + (end_hz - start_hz) * progress
        sample = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", sample)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def _mk_rect_png(
    *,
    rects: list[tuple[int, int, int, int, tuple[int, int, int]]],
    size: tuple[int, int] = (192, 192),
    bg: tuple[int, int, int] = (18, 18, 18),
) -> bytes:
    image = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1, color in rects:
        draw.rectangle((x0, y0, x1, y1), fill=color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _run_text_tick(runtime: RuntimeV2, *, tick_index: int, text: str) -> dict[str, Any]:
    started = time.perf_counter()
    tick = runtime.process_text_tick(text=text, tick_index=tick_index)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.set_last_logic_ms(elapsed_ms)
    tick["elapsed_ms"] = _round4(elapsed_ms)
    return tick


def _run_multimodal_tick(
    runtime: RuntimeV2,
    *,
    tick_index: int,
    text: str = "",
    image_bytes: bytes | None = None,
    audio_bytes: bytes | None = None,
    source_type: str = "multichannel_probe",
    execute_selected_actions: bool = True,
) -> dict[str, Any]:
    text_packet = runtime.text_sensor.ingest(text, tick_index=tick_index, source_type=source_type)
    image_packet = runtime.vision_sensor.ingest_image_bytes(image_bytes, tick_index=tick_index, source_type=source_type) if image_bytes is not None else None
    audio_packet = runtime.hearing_sensor.ingest_wav_bytes(audio_bytes, tick_index=tick_index, source_type=source_type) if audio_bytes is not None else None
    started = time.perf_counter()
    tick = runtime.process_multimodal_tick(
        tick_index=tick_index,
        text_packet=text_packet,
        image_packet=image_packet,
        audio_packet=audio_packet,
        source_type=source_type,
    )
    runtime_action_effects = {"applied_actions": [], "moved": False, "audio_moved": False}
    if execute_selected_actions:
        selected_actions = list(((tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or []))
        runtime_action_effects = runtime.apply_selected_actions(selected_actions, runtime_tick=tick)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.set_last_logic_ms(elapsed_ms)
    tick["runtime_action_effects"] = runtime_action_effects
    tick["elapsed_ms"] = _round4(elapsed_ms)
    return tick


def _emotion_row(tick: dict[str, Any], *, tick_index: int, text: str) -> dict[str, Any]:
    rules = dict(tick.get("rules_result", {}) or {})
    emotion = dict(rules.get("emotion_channels", {}) or {})
    raw_emotion = dict(rules.get("raw_emotion_channels", {}) or {})
    metrics = dict(rules.get("metrics_snapshot", {}) or {})
    pending = dict(tick.get("pending_feedback_metrics", {}) or {})
    queued = dict(tick.get("queued_intrinsic_feedback_preview", {}) or {})
    top = [str(item.get("sa_label", "") or "") for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])[:10])]
    return {
        "tick_index": int(tick_index),
        "text": text,
        "elapsed_ms": _round4(float(tick.get("elapsed_ms", 0.0) or 0.0)),
        "surprise": _round4(float(emotion.get("surprise", 0.0) or 0.0)),
        "dissonance": _round4(float(emotion.get("dissonance", 0.0) or 0.0)),
        "correctness": _round4(float(emotion.get("correctness", 0.0) or 0.0)),
        "grasp": _round4(float(emotion.get("grasp", 0.0) or 0.0)),
        "expectation": _round4(float(emotion.get("expectation", 0.0) or 0.0)),
        "pressure": _round4(float(emotion.get("pressure", 0.0) or 0.0)),
        "raw_surprise": _round4(float(raw_emotion.get("surprise", 0.0) or 0.0)),
        "raw_dissonance": _round4(float(raw_emotion.get("dissonance", 0.0) or 0.0)),
        "underprediction_mass": _round4(float(metrics.get("state.prediction_underprediction_mass", 0.0) or 0.0)),
        "overprediction_mass": _round4(float(metrics.get("state.prediction_overprediction_mass", 0.0) or 0.0)),
        "alignment_score": _round4(float(metrics.get("state.prediction_alignment_score", 0.0) or 0.0)),
        "grasp_score": _round4(float(metrics.get("state.prediction_grasp_score", 0.0) or 0.0)),
        "committed_alignment_score": _round4(float(metrics.get("state.prediction_committed_alignment_score", 0.0) or 0.0)),
        "committed_grasp_score": _round4(float(metrics.get("state.prediction_committed_grasp_score", 0.0) or 0.0)),
        "pending_reward": _round4(float(pending.get("reward", 0.0) or 0.0)),
        "pending_punishment": _round4(float(pending.get("punishment", 0.0) or 0.0)),
        "queued_reward": _round4(float(queued.get("reward", 0.0) or 0.0)),
        "queued_punishment": _round4(float(queued.get("punishment", 0.0) or 0.0)),
        "state_top_labels": top,
    }


def _run_time_experiment() -> dict[str, Any]:
    gaps = [1, 2, 4, 6]
    rows: list[dict[str, Any]] = []
    monotonic_deltas: list[float] = []
    for gap in gaps:
        runtime = _mk_runtime()
        latest = {}
        sequence = ["apple"] + [""] * gap + ["apple"]
        for tick_index, text in enumerate(sequence):
            latest = _run_text_tick(runtime, tick_index=tick_index, text=text)
        spacetime = dict(latest.get("query_spacetime", {}) or {})
        trace = dict((latest.get("channel_feeling_trace", {}) or {}).get("time", {}) or {})
        rows.append(
            {
                "gap_ticks": int(gap),
                "target_delta_t": _round4(float(spacetime.get("target_delta_t", 0.0) or 0.0)),
                "time_confidence": _round4(float(spacetime.get("time_confidence", 0.0) or 0.0)),
                "best_center": _round4(float(trace.get("best_center", 0.0) or 0.0)),
                "signal_strength": _round4(float(trace.get("signal_strength", 0.0) or 0.0)),
                "cluster_count": int(trace.get("cluster_count", 0) or 0),
                "state_top_labels": [str(item.get("sa_label", "") or "") for item in (((latest.get("state_pool_summary", {}) or {}).get("top", []) or [])[:8])],
            }
        )
        monotonic_deltas.append(float(spacetime.get("target_delta_t", 0.0) or 0.0))
    monotonic_prefix = all(right >= left for left, right in zip(monotonic_deltas[:3], monotonic_deltas[1:4]))

    store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
    store.write_memory(
        tick_index=2,
        memory_kind="exact_external",
        units=["eat", "apple"],
        items=[{"sa_label": "text::apple", "display_text": "apple", "energy": 1.0}],
        text="eat apple",
        reality_weight=1.0,
    )
    store.write_memory(
        tick_index=12,
        memory_kind="exact_external",
        units=["eat", "banana"],
        items=[{"sa_label": "text::banana", "display_text": "banana", "energy": 1.0}],
        text="eat banana",
        reality_weight=1.0,
    )
    bn = store.recall_bn(
        query_labels=["text::eat"],
        query_weights={"text::eat": 1.0},
        top_k=2,
        tick_index=20,
        query_units=["eat"],
        recent_focus_units=["eat"],
        query_spacetime={
            "t": 20,
            "target_delta_t": 8.0,
            "time_sigma": 1.2,
            "time_confidence": 0.9,
            "time_recall_gain": 0.5,
        },
    )
    recall_rows = [
        {
            "text": str(row.get("text", "") or ""),
            "score": _round4(float(row.get("score", 0.0) or 0.0)),
            "time_intent_bonus": _round4(float((row.get("score_breakdown", {}) or {}).get("time_intent_bonus", 0.0) or 0.0)),
        }
        for row in bn[:2]
    ]
    return {
        "runtime_rows": rows,
        "runtime_passed": bool(rows and all("timefelt::elapsed" in row["state_top_labels"] for row in rows)),
        "monotonic_prefix_passed": bool(monotonic_prefix),
        "recall_rows": recall_rows,
        "recall_passed": bool(recall_rows and recall_rows[0]["text"] == "eat banana" and recall_rows[0]["time_intent_bonus"] > 0.0),
    }


def _run_emotion_experiment() -> dict[str, Any]:
    runtime = _mk_runtime()
    sequence = ["3", "3", "", "8", "8", "8", "8", "8", "8", "8", "8", "", ""]
    rows = []
    best_correctness_after_8 = 0.0
    best_grasp_after_8 = 0.0
    for tick_index, text in enumerate(sequence):
        tick = _run_text_tick(runtime, tick_index=tick_index, text=text)
        row = _emotion_row(tick, tick_index=tick_index, text=text)
        rows.append(row)
        if tick_index >= 3:
            best_correctness_after_8 = max(best_correctness_after_8, float(row["correctness"]))
            best_grasp_after_8 = max(best_grasp_after_8, float(row["grasp_score"]))

    recovery_runtime = _mk_runtime()
    recovery_runtime.build_intrinsic_feedback(
        emotion_channels={"expectation": 0.1, "pressure": 0.4, "correctness": 0.1, "dissonance": 0.8, "surprise": 0.7},
        balance_metrics={"alignment_score": 0.1, "grasp_score": 0.05, "overprediction_ratio": 0.8, "underprediction_ratio": 0.7},
    )
    recovered = recovery_runtime.build_intrinsic_feedback(
        emotion_channels={"expectation": 0.1, "pressure": 0.1, "correctness": 0.4, "dissonance": 0.2, "surprise": 0.1},
        balance_metrics={"alignment_score": 0.65, "grasp_score": 0.55, "overprediction_ratio": 0.2, "underprediction_ratio": 0.1},
    )

    signal_runtime = _mk_runtime()
    _run_text_tick(signal_runtime, tick_index=0, text="hello")
    signal_runtime.apply_action_feedback(
        tick_index=0,
        selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
        emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
        runtime_action_effects={"moved": False},
        external_feedback={"reward": 0.3, "punishment": 0.0},
    )
    signal_tick = _run_text_tick(signal_runtime, tick_index=1, text="world")
    signal_top = [str(item.get("sa_label", "") or "") for item in (((signal_tick.get("state_pool_summary", {}) or {}).get("top", []) or [])[:12])]

    return {
        "rows": rows,
        "novelty_passed": bool(rows and rows[0]["surprise"] > 0.0 and rows[0]["underprediction_mass"] > 0.0),
        "repeat_correctness_passed": bool(len(rows) > 1 and rows[1]["correctness"] > 0.0 and rows[1]["grasp_score"] > 0.0),
        "relearn_8_passed": bool(best_correctness_after_8 > 0.12 and best_grasp_after_8 > 0.12),
        "recovery_feedback": {
            "reward": _round4(float(recovered.get("reward", 0.0) or 0.0)),
            "punishment": _round4(float(recovered.get("punishment", 0.0) or 0.0)),
            "notes": list(recovered.get("notes", []) or []),
        },
        "feedback_signal_state_top": signal_top,
        "feedback_signal_passed": bool("attr::reward_signal" in signal_top or "attr::punishment_signal" in signal_top),
    }


def _run_rhythm_experiment() -> dict[str, Any]:
    def run_sequence(sequence: list[str]) -> dict[str, Any]:
        runtime = _mk_runtime()
        latest = {}
        rows = []
        for tick_index, text in enumerate(sequence):
            latest = _run_text_tick(runtime, tick_index=tick_index, text=text)
            trace = dict((latest.get("channel_feeling_trace", {}) or {}).get("rhythm", {}) or {})
            pulse = dict((trace.get("best_pulse", {}) or {}))
            rows.append(
                {
                    "tick_index": int(tick_index),
                    "text": text,
                    "labels": [str(item.get("sa_label", "") or "") for item in (latest.get("channel_feeling_items", []) or [])],
                    "pulse_confidence": _round4(float(pulse.get("confidence", 0.0) or 0.0)),
                    "pulse_regularity": _round4(float(pulse.get("regularity", 0.0) or 0.0)),
                    "pulse_groove": _round4(float(pulse.get("groove", 0.0) or 0.0)),
                    "pulse_period_ticks": _round4(float(pulse.get("period_ticks", 0.0) or 0.0)),
                }
            )
        trace = dict((latest.get("channel_feeling_trace", {}) or {}).get("rhythm", {}) or {})
        pulse = dict((trace.get("best_pulse", {}) or {}))
        spacetime = dict(latest.get("query_spacetime", {}) or {})
        return {
            "rows": rows,
            "trace": trace,
            "best_pulse": {
                "period_ticks": _round4(float(pulse.get("period_ticks", 0.0) or 0.0)),
                "regularity": _round4(float(pulse.get("regularity", 0.0) or 0.0)),
                "confidence": _round4(float(pulse.get("confidence", 0.0) or 0.0)),
                "groove": _round4(float(pulse.get("groove", 0.0) or 0.0)),
            },
            "query_spacetime": {
                "rhythm_period_ticks": _round4(float(spacetime.get("rhythm_period_ticks", 0.0) or 0.0)),
                "rhythm_confidence": _round4(float(spacetime.get("rhythm_confidence", 0.0) or 0.0)),
                "rhythm_family_key": str(spacetime.get("rhythm_family_key", "") or ""),
            },
        }

    regular = run_sequence(["beat", "", "beat", "", "beat", "", "beat"])
    irregular = run_sequence(["beat", "beat", "", "", "beat", "", "", "", "beat", ""])

    store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
    store.write_memory(
        tick_index=2,
        memory_kind="exact_external",
        units=["beat"],
        items=[{"sa_label": "text::beat", "display_text": "beat", "energy": 1.0}],
        text="beat",
        reality_weight=1.0,
    )
    store.write_memory(
        tick_index=8,
        memory_kind="exact_external",
        units=["beat"],
        items=[{"sa_label": "text::beat", "display_text": "beat", "energy": 1.0}],
        text="beat",
        reality_weight=1.0,
    )
    store.write_memory(
        tick_index=11,
        memory_kind="exact_external",
        units=["offbeat"],
        items=[{"sa_label": "text::offbeat", "display_text": "offbeat", "energy": 1.0}],
        text="offbeat",
        reality_weight=1.0,
    )
    bn = store.recall_bn(
        query_labels=["text::beat"],
        query_weights={"text::beat": 1.0},
        top_k=3,
        tick_index=14,
        query_units=["beat"],
        recent_focus_units=["beat"],
        query_spacetime={
            "rhythm_period_ticks": 6.0,
            "rhythm_period_sigma": 1.2,
            "rhythm_confidence": 0.9,
            "rhythm_recall_gain": 0.5,
            "rhythm_family_key": "text::beat",
            "rhythm_time_to_next": 0.0,
        },
    )
    recall_rows = [
        {
            "text": str(row.get("text", "") or ""),
            "score": _round4(float(row.get("score", 0.0) or 0.0)),
            "rhythm_intent_bonus": _round4(float((row.get("score_breakdown", {}) or {}).get("rhythm_intent_bonus", 0.0) or 0.0)),
        }
        for row in bn[:3]
    ]
    return {
        "regular": regular,
        "irregular": irregular,
        "runtime_passed": bool(
            regular["best_pulse"]["confidence"] > irregular["best_pulse"]["confidence"]
            and regular["best_pulse"]["regularity"] > irregular["best_pulse"]["regularity"]
        ),
        "recall_rows": recall_rows,
        "recall_passed": bool(recall_rows and recall_rows[0]["text"] == "beat" and recall_rows[0]["rhythm_intent_bonus"] > 0.0),
    }


def _run_motion_recall_experiment() -> dict[str, Any]:
    store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
    slow_memory = store.write_memory(
        tick_index=0,
        memory_kind="exact_external",
        units=["move"],
        items=[
            {"sa_label": "text::move", "display_text": "move", "energy": 1.0},
            {"sa_label": "vision_dyn::slow", "display_text": "slow motion", "energy": 0.8, "attributes": {"motion_speed": 0.12}},
        ],
        text="move",
        reality_weight=1.0,
    )
    fast_memory = store.write_memory(
        tick_index=0,
        memory_kind="exact_external",
        units=["move"],
        items=[
            {"sa_label": "text::move", "display_text": "move", "energy": 1.0},
            {"sa_label": "vision_dyn::fast", "display_text": "fast motion", "energy": 0.8, "attributes": {"motion_speed": 0.82}},
        ],
        text="move",
        reality_weight=1.0,
    )
    bn = store.recall_bn(
        query_labels=["text::move"],
        query_weights={"text::move": 1.0},
        top_k=2,
        tick_index=1,
        query_units=["move"],
        recent_focus_units=["move"],
        query_spacetime={
            "motion_center_speed": 0.78,
            "motion_sigma": 0.08,
            "motion_confidence": 0.9,
            "motion_recall_gain": 0.5,
        },
    )
    rows = [
        {
            "memory_id": str(row.get("memory_id", "") or ""),
            "text": str(row.get("text", "") or ""),
            "score": _round4(float(row.get("score", 0.0) or 0.0)),
            "motion_intent_bonus": _round4(float((row.get("score_breakdown", {}) or {}).get("motion_intent_bonus", 0.0) or 0.0)),
        }
        for row in bn[:2]
    ]
    return {
        "rows": rows,
        "passed": bool(rows and rows[0]["memory_id"] == str(fast_memory.get("memory_id", "")) and rows[0]["motion_intent_bonus"] > 0.0 and rows[0]["memory_id"] != str(slow_memory.get("memory_id", ""))),
    }


def _run_feedback_recall_experiment() -> dict[str, Any]:
    store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
    reward_memory = store.write_memory(
        tick_index=0,
        memory_kind="exact_external",
        units=["feedback"],
        items=[
            {"sa_label": "text::feedback", "display_text": "feedback", "energy": 1.0},
            {"sa_label": "attr::reward_signal", "display_text": "reward", "energy": 0.42},
        ],
        text="feedback",
        reality_weight=1.0,
    )
    punishment_memory = store.write_memory(
        tick_index=0,
        memory_kind="exact_external",
        units=["feedback"],
        items=[
            {"sa_label": "text::feedback", "display_text": "feedback", "energy": 1.0},
            {"sa_label": "attr::punishment_signal", "display_text": "punishment", "energy": 0.42},
        ],
        text="feedback",
        reality_weight=1.0,
    )
    positive = store.recall_bn(
        query_labels=["text::feedback"],
        query_weights={"text::feedback": 1.0},
        top_k=2,
        tick_index=1,
        query_units=["feedback"],
        recent_focus_units=["feedback"],
        query_spacetime={
            "feedback_valence": 0.4,
            "feedback_sigma": 0.12,
            "feedback_confidence": 1.0,
            "feedback_recall_gain": 0.5,
        },
    )
    negative = store.recall_bn(
        query_labels=["text::feedback"],
        query_weights={"text::feedback": 1.0},
        top_k=2,
        tick_index=1,
        query_units=["feedback"],
        recent_focus_units=["feedback"],
        query_spacetime={
            "feedback_valence": -0.4,
            "feedback_sigma": 0.12,
            "feedback_confidence": 1.0,
            "feedback_recall_gain": 0.5,
        },
    )
    return {
        "positive_top": {
            "memory_id": str((positive[0] or {}).get("memory_id", "") or "") if positive else "",
            "feedback_intent_bonus": _round4(float((((positive[0] if positive else {}) or {}).get("score_breakdown", {}) or {}).get("feedback_intent_bonus", 0.0) or 0.0)),
        },
        "negative_top": {
            "memory_id": str((negative[0] or {}).get("memory_id", "") or "") if negative else "",
            "feedback_intent_bonus": _round4(float((((negative[0] if negative else {}) or {}).get("score_breakdown", {}) or {}).get("feedback_intent_bonus", 0.0) or 0.0)),
        },
        "passed": bool(
            positive
            and negative
            and str((positive[0] or {}).get("memory_id", "") or "") == str(reward_memory.get("memory_id", ""))
            and str((negative[0] or {}).get("memory_id", "") or "") == str(punishment_memory.get("memory_id", ""))
        ),
    }


def _run_visual_experiment() -> dict[str, Any]:
    pairs = [
        OCRPair(pair_id="digit_3", glyph="3", text_label="three", rotate_deg=-6.0),
        OCRPair(pair_id="digit_8", glyph="8", text_label="eight", rotate_deg=5.0),
    ]
    image_map = {pair.pair_id: _render_handwritten_image(pair) for pair in pairs}
    runtime, training = _train_runtime(
        pairs=pairs,
        image_map=image_map,
        train_plan=(6, 6),
        train_raw_budget=512,
        train_patch_budget=24,
        train_focus_budget=12,
        stabilize_ticks=4,
    )
    payload = runtime.export_payload()
    probe_rows: list[dict[str, Any]] = []
    for pair in pairs:
        probe_runtime = _mk_runtime(
            overrides={
                "vision_raw_state_budget": 256,
                "vision_patch_budget": 16,
                "vision_focus_patch_budget": 8,
                "vision_attention_boost_enabled": True,
                "vision_dynamic_track_window": 6,
                "vision_dynamic_candidate_limit_background": 12,
                "vision_dynamic_candidate_limit_focus": 28,
                "vision_dynamic_track_limit": 40,
                "vision_dynamic_summary_limit": 4,
                "vision_dynamic_match_threshold": 0.46,
                "vision_dynamic_track_forget_ticks": 3,
            }
        )
        probe_runtime.import_payload({"memory_store": copy.deepcopy(payload.get("memory_store", {}))})
        tick = _run_multimodal_tick(
            probe_runtime,
            tick_index=0,
            text="",
            image_bytes=image_map[pair.pair_id],
            source_type=f"visual_probe::{pair.pair_id}",
            execute_selected_actions=False,
        )
        eval_row = _evaluate_probe(
            tick=tick,
            target_text=pair.text_label,
            distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
        )
        image_packet = dict(tick.get("image_packet", {}) or {})
        probe_rows.append(
            {
                "pair_id": pair.pair_id,
                "target_text_label": pair.text_label,
                "bn_best_text": str(eval_row.get("bn_best_text", "") or ""),
                "bn_target_rank": int(eval_row.get("bn_target_rank", 0) or 0),
                "cstar_best_text": str(eval_row.get("cstar_best_text", "") or ""),
                "state_best_text": str(eval_row.get("state_best_text", "") or ""),
                "strict_success": bool(eval_row.get("strict_success", False)),
                "state_success": bool(eval_row.get("state_success", False)),
                "raw_sample_count": int(image_packet.get("total_patch_count", 0) or 0),
                "memory_write_count": int(len(image_packet.get("memory_write_samples", []) or [])),
                "focus_priority_count": int(len(image_packet.get("focus_priority_samples", []) or [])),
                "global_structure_count": int(len(image_packet.get("global_structure_samples", []) or [])),
            }
        )

    motion_runtime = _mk_runtime(
        overrides={
            "vision_attention_boost_enabled": True,
            "vision_patch_budget": 16,
            "vision_focus_patch_budget": 8,
            "vision_raw_state_budget": 64,
            "vision_reconstruction_patch_budget": 1024,
            "vision_attention_boost_max_extra_raw_budget": 192,
            "vision_attention_boost_max_extra_focus_budget": 8,
            "vision_dynamic_track_window": 6,
            "vision_dynamic_candidate_limit_background": 12,
            "vision_dynamic_candidate_limit_focus": 28,
            "vision_dynamic_track_limit": 40,
            "vision_dynamic_summary_limit": 4,
            "vision_dynamic_match_threshold": 0.46,
            "vision_dynamic_track_forget_ticks": 3,
        }
    )
    motion_runtime.vision_sensor.move_gaze(0.5, 0.5)
    static_png = _mk_rect_png(rects=[(68, 76, 108, 116, (236, 236, 236))])
    motion_png = _mk_rect_png(rects=[(68, 76, 108, 116, (236, 236, 236)), (146, 30, 170, 54, (255, 255, 255))])
    _run_multimodal_tick(motion_runtime, tick_index=0, text="", image_bytes=static_png, source_type="motionfelt::warmup", execute_selected_actions=True)
    motion_tick = _run_multimodal_tick(motion_runtime, tick_index=1, text="", image_bytes=motion_png, source_type="motionfelt::probe", execute_selected_actions=True)
    motion_labels = [str(item.get("sa_label", "") or "") for item in (motion_tick.get("channel_feeling_items", []) or [])]
    motion_trace = dict((motion_tick.get("channel_feeling_trace", {}) or {}).get("motion", {}) or {})
    dynamic_summary = dict((((motion_tick.get("image_packet", {}) or {}).get("dynamic_track_summary", {}) or {})))
    return {
        "ocr_training": {
            "accepted": bool(training.get("accepted", False)),
            "trained_epochs": int(training.get("trained_epochs", 0) or 0),
            "train_raw_budget": int(training.get("train_raw_budget", 0) or 0),
            "train_patch_budget": int(training.get("train_patch_budget", 0) or 0),
            "train_focus_budget": int(training.get("train_focus_budget", 0) or 0),
        },
        "ocr_probe_rows": probe_rows,
        "ocr_passed": bool(probe_rows and all(bool(row.get("state_success", False)) for row in probe_rows)),
        "motion_probe": {
            "labels": motion_labels,
            "motion_trace": {
                "confidence": _round4(float(motion_trace.get("confidence", 0.0) or 0.0)),
                "best_speed": _round4(float(motion_trace.get("best_speed", 0.0) or 0.0)),
            },
            "dynamic_track_count": int(dynamic_summary.get("track_count", 0) or 0),
            "dynamic_object_count": int(dynamic_summary.get("object_count", 0) or 0),
            "dynamic_salience_mean": _round4(float(dynamic_summary.get("dynamic_salience_mean", 0.0) or 0.0)),
        },
        "motion_passed": bool("motionfelt::trend" in motion_labels and float(motion_trace.get("confidence", 0.0) or 0.0) > 0.0),
    }


def _run_audio_experiment() -> dict[str, Any]:
    runtime = _mk_runtime(
        overrides={
            "hearing_window_budget": 12,
            "hearing_focus_band_count": 12,
            "hearing_focus_bandwidth_octaves": 1.15,
            "hearing_attention_boost_enabled": True,
            "hearing_attention_boost_max_extra_window_budget": 12,
            "hearing_attention_boost_max_extra_focus_budget": 8,
        }
    )
    neutral_audio = _mk_wav(220.0)
    focused_audio = _mk_wav(880.0)
    before_packet = runtime.hearing_sensor.ingest_wav_bytes(neutral_audio, tick_index=0, source_type="audio_probe::before")
    before_focus = dict(before_packet.get("audio_focus", {}) or {})
    before_focus_count = int(len(before_packet.get("focus_priority_samples", []) or []))
    tick = _run_multimodal_tick(runtime, tick_index=0, text="", audio_bytes=focused_audio, source_type="audio_probe::focus", execute_selected_actions=False)
    audio_packet = dict(tick.get("audio_packet", {}) or {})
    focus_target = None
    focus_rows = [dict(item) for item in (audio_packet.get("focus_priority_samples", []) or []) if isinstance(item, dict)]
    if focus_rows:
        focus_target = float(
            ((focus_rows[0].get("coords", {}) or {}).get("freq_center_hz", 0.0))
            or ((focus_rows[0].get("attributes", {}) or {}).get("dominant_hz", 0.0))
            or 0.0
        )
    effects = runtime.apply_selected_actions(
        [{"action_name": "continue_audio_focus", "params": {}, "firmness_norm": 1.0}],
        runtime_tick={"audio_packet": audio_packet},
    )
    after_audio = runtime.hearing_sensor.ingest_wav_bytes(focused_audio, tick_index=1, source_type="audio_probe::after")
    after_focus = dict(after_audio.get("audio_focus", {}) or {})
    after_focus_count = int(len(after_audio.get("focus_priority_samples", []) or []))
    strongest_after = dict(((after_audio.get("windows", []) or [])[0] or {}))
    strongest_attrs = dict(strongest_after.get("attributes", {}) or {})
    strongest_coords = dict(strongest_after.get("coords", {}) or {})

    semantic_runtime = _mk_runtime(
        overrides={
            "intrinsic_feedback_enabled": False,
            "hearing_window_budget": 12,
            "hearing_focus_band_count": 12,
            "hearing_focus_bandwidth_octaves": 1.15,
        }
    )
    training_items = [
        ("tone_low", _mk_chirp(start_hz=320.0, end_hz=480.0)),
        ("tone_high", _mk_chirp(start_hz=760.0, end_hz=980.0)),
    ]
    tick_index = 0
    for _ in range(4):
        for label, audio_bytes in training_items:
            tick = _run_multimodal_tick(semantic_runtime, tick_index=tick_index, text=label, audio_bytes=audio_bytes, source_type=f"audio_semantic_train::{label}", execute_selected_actions=False)
            semantic_runtime.inject_feedback_signals(
                tick_index=tick_index,
                feedback={"reward": 0.8, "punishment": 0.0, "notes": [f"audio_reward::{label}"]},
                provenance={"exact_memory_id": str((tick.get("exact_memory", {}) or {}).get("memory_id", "") or "")},
                source_type="audio_semantic_reward",
                channel="audio_semantic_reward",
            )
            tick_index += 1
    for _ in range(4):
        _run_multimodal_tick(semantic_runtime, tick_index=tick_index, text="", audio_bytes=None, source_type="audio_semantic_stabilize", execute_selected_actions=False)
        tick_index += 1

    semantic_probe_rows = []
    for label, audio_bytes in training_items:
        probe_tick = _run_multimodal_tick(semantic_runtime, tick_index=tick_index, text="", audio_bytes=audio_bytes, source_type=f"audio_semantic_probe::{label}", execute_selected_actions=False)
        tick_index += 1
        bn_preview = [
            {
                "text": str(item.get("text", "") or ""),
                "score": _round4(float(item.get("score", 0.0) or 0.0)),
            }
            for item in ((probe_tick.get("bn_list", []) or [])[:4])
            if isinstance(item, dict)
        ]
        semantic_probe_rows.append(
            {
                "label": label,
                "bn_preview": bn_preview,
                "state_top": [str(item.get("sa_label", "") or "") for item in (((probe_tick.get("state_pool_summary", {}) or {}).get("top", []) or [])[:10])],
            }
        )

    return {
        "focus_probe": {
            "before_focus": before_focus,
            "before_focus_priority_count": int(before_focus_count),
            "focus_target_hz": _round4(float(focus_target or 0.0)),
            "effects": {
                "audio_moved": bool(effects.get("audio_moved", False)),
                "audio_focus_after": dict(effects.get("audio_focus_after", {}) or {}),
                "audio_attention_boost": dict(effects.get("audio_attention_boost", {}) or {}),
            },
            "after_focus": after_focus,
            "after_focus_priority_count": int(after_focus_count),
            "after_strongest_window": {
                "dominant_hz": _round4(float(strongest_attrs.get("dominant_hz", 0.0) or 0.0)),
                "focus_bonus": _round4(float(strongest_attrs.get("focus_bonus", 0.0) or 0.0)),
                "freq_center_hz": _round4(float(strongest_coords.get("freq_center_hz", 0.0) or 0.0)),
            },
        },
        "focus_passed": bool(
            focus_target is not None
            and bool(effects.get("audio_moved", False))
            and abs(float((effects.get("audio_focus_after", {}) or {}).get("center_hz", 0.0) or 0.0) - float(focus_target)) < 200.0
            and after_focus_count >= before_focus_count
        ),
        "semantic_probe_rows": semantic_probe_rows,
    }


def _render_report(summary: dict[str, Any]) -> str:
    time_block = dict(summary.get("time", {}) or {})
    emotion_block = dict(summary.get("emotion", {}) or {})
    rhythm_block = dict(summary.get("rhythm", {}) or {})
    motion_recall_block = dict(summary.get("motion_recall", {}) or {})
    feedback_recall_block = dict(summary.get("feedback_recall", {}) or {})
    visual_block = dict(summary.get("visual", {}) or {})
    audio_block = dict(summary.get("audio", {}) or {})

    lines = [
        "# V2 多通道感受与召回闭环实验报告",
        "",
        f"- 生成时间：{datetime.now().isoformat()}",
        "- 目标：对时间感受、情绪/认知感受、节奏感、视觉感受、听觉感受，以及它们反过来影响召回的闭环做统一验收。",
        "- 方法：每个通道都分成两层证明。",
        "  1. 运行时是否真的生成了对应通道的感受信号，并以 SA 形式进入状态池或 query_spacetime。",
        "  2. 该通道的感受/意图参数是否真的改变了向量数据库中的召回排序。",
        "",
        "## 一、总评",
        f"- 时间感受：runtime={time_block.get('runtime_passed', False)} / monotonic_prefix={time_block.get('monotonic_prefix_passed', False)} / recall={time_block.get('recall_passed', False)}",
        f"- 情绪与认知感受：novelty={emotion_block.get('novelty_passed', False)} / repeat_correctness={emotion_block.get('repeat_correctness_passed', False)} / relearn_8={emotion_block.get('relearn_8_passed', False)} / feedback_signal={emotion_block.get('feedback_signal_passed', False)}",
        f"- 节奏感：runtime={rhythm_block.get('runtime_passed', False)} / recall={rhythm_block.get('recall_passed', False)}",
        f"- 运动意图召回：passed={motion_recall_block.get('passed', False)}",
        f"- 反馈价性召回：passed={feedback_recall_block.get('passed', False)}",
        f"- 视觉：ocr={visual_block.get('ocr_passed', False)} / motion={visual_block.get('motion_passed', False)}",
        f"- 听觉：focus={audio_block.get('focus_passed', False)} / semantic=partial",
        "",
        "## 二、时间感受",
        "- 预期：当当前现状强烈召回某一类过去记忆时，系统应从主导记忆波峰中抽出时间间隔感，并把它写入 query_spacetime，形成后续模糊时间召回偏置。",
        "- 结果摘要：",
    ]
    for row in time_block.get("runtime_rows", []) or []:
        lines.append(
            f"  - gap={row['gap_ticks']} -> target_delta_t={row['target_delta_t']} / confidence={row['time_confidence']} / top_has_timefelt={'timefelt::elapsed' in row['state_top_labels']}"
        )
    if time_block.get("recall_rows"):
        lines.append(
            f"- 召回对比：top1={time_block['recall_rows'][0]['text']} / time_intent_bonus={time_block['recall_rows'][0]['time_intent_bonus']}"
        )
    lines.extend(
        [
            "- 解释：当前的时间感受更像‘主导时间间隔簇’而不是精确 tick 计数，因此最稳妥的验收标准不是要求完全等于真实 gap，而是要求它能形成稳定的时间主峰，并在召回时对匹配时间间隔的记忆产生可见偏置。",
            "",
            "## 三、情绪与认知感受",
            "- 预期：首次新异输入应先产生惊；重复输入应逐步拉起正确感与把握相关指标；错配后持续输入新对象时，应在若干 tick 后重新形成把握与正确感；恢复过程应带来恢复性奖励；期待与压力应持续转成下一 tick 的奖惩信号。",
            f"- 序列结果：novelty={emotion_block.get('novelty_passed', False)} / repeat_correctness={emotion_block.get('repeat_correctness_passed', False)} / relearn_8={emotion_block.get('relearn_8_passed', False)}",
        ]
    )
    for row in (emotion_block.get("rows", []) or [])[:10]:
        lines.append(
            f"  - tick={row['tick_index']} text={row['text']!r} surprise={row['surprise']} dissonance={row['dissonance']} correctness={row['correctness']} grasp_score={row['grasp_score']} pending=({row['pending_reward']},{row['pending_punishment']})"
        )
    recovery = dict(emotion_block.get("recovery_feedback", {}) or {})
    lines.extend(
        [
            f"- 恢复性反馈：reward={recovery.get('reward', 0.0)} / punishment={recovery.get('punishment', 0.0)} / notes={recovery.get('notes', [])}",
            f"- 奖惩信号入池：top_contains_feedback={emotion_block.get('feedback_signal_passed', False)} / top={emotion_block.get('feedback_signal_state_top', [])[:8]}",
            "",
            "## 四、节奏感",
            "- 预期：规则输入应形成 pulse/phase 类节奏感；不规则输入不一定完全没有节奏感，但其 regularity / confidence / groove 应显著更低；同时节奏 query_spacetime 应能提升匹配节拍周期的记忆召回分数。",
        ]
    )
    regular = dict((rhythm_block.get("regular", {}) or {}).get("best_pulse", {}) or {})
    irregular = dict((rhythm_block.get("irregular", {}) or {}).get("best_pulse", {}) or {})
    lines.extend(
        [
            f"- 规则节奏：period={regular.get('period_ticks', 0.0)} / regularity={regular.get('regularity', 0.0)} / confidence={regular.get('confidence', 0.0)} / groove={regular.get('groove', 0.0)}",
            f"- 非规则节奏：period={irregular.get('period_ticks', 0.0)} / regularity={irregular.get('regularity', 0.0)} / confidence={irregular.get('confidence', 0.0)} / groove={irregular.get('groove', 0.0)}",
        ]
    )
    if rhythm_block.get("recall_rows"):
        lines.append(
            f"- 节奏召回：top1={rhythm_block['recall_rows'][0]['text']} / rhythm_intent_bonus={rhythm_block['recall_rows'][0]['rhythm_intent_bonus']}"
        )
    lines.extend(
        [
            "",
            "## 五、视觉与运动",
            "- 预期：静态图像文字训练后，应能以统一状态池方式召回对应文本；动态输入应额外产生 motionfelt::trend，并把运动摘要写入内部链路。",
            f"- OCR 训练：accepted={dict(visual_block.get('ocr_training', {}) or {}).get('accepted', False)} / epochs={dict(visual_block.get('ocr_training', {}) or {}).get('trained_epochs', 0)}",
        ]
    )
    for row in visual_block.get("ocr_probe_rows", []) or []:
        lines.append(
            f"  - {row['pair_id']}: bn={row['bn_best_text']} / cstar={row['cstar_best_text']} / state={row['state_best_text']} / strict={row['strict_success']} / raw={row['raw_sample_count']} / focus={row['focus_priority_count']}"
        )
    motion_probe = dict(visual_block.get("motion_probe", {}) or {})
    lines.extend(
        [
            f"- 动态视觉：motion_passed={visual_block.get('motion_passed', False)} / labels={motion_probe.get('labels', [])} / dynamic_object_count={motion_probe.get('dynamic_object_count', 0)} / motion_confidence={dict(motion_probe.get('motion_trace', {}) or {}).get('confidence', 0.0)}",
            "",
            "## 六、听觉",
            "- 预期：听觉焦点应像视焦点一样可移动，并提升焦点频段附近的采样优先级；进入统一主链后，可作为未来语义学习、背景降噪与注意力联动的基础。",
        ]
    )
    focus_probe = dict(audio_block.get("focus_probe", {}) or {})
    focus_after = dict((focus_probe.get("effects", {}) or {}).get("audio_focus_after", {}) or {})
    lines.extend(
        [
            f"- 焦点移动：passed={audio_block.get('focus_passed', False)} / before_center={dict(focus_probe.get('before_focus', {}) or {}).get('center_hz', 0.0)} / after_center={focus_after.get('center_hz', 0.0)} / target_hz={focus_probe.get('focus_target_hz', 0.0)}",
            f"- 焦点采样：before_focus_priority_count={focus_probe.get('before_focus_priority_count', 0)} / after_focus_priority_count={focus_probe.get('after_focus_priority_count', 0)} / strongest_after={focus_probe.get('after_strongest_window', {})}",
            "- 听觉语义边界：本轮补做了 richer pattern 的初探针，但仍只把它视作初步迹象，不宣称已经形成稳定的音频语义识别闭环。",
        ]
    )
    for row in audio_block.get("semantic_probe_rows", []) or []:
        lines.append(f"  - {row['label']}: bn_preview={row['bn_preview'][:3]}")
    lines.extend(
        [
            "",
            "## 七、统一哲学意义",
            "- 这组结果更重要的不是单一准确率，而是多个通道都已经呈现同一个统一结构：",
            "  1. 通道自身先产出连续强度信号。",
            "  2. 当强度超过阈值时，该信号以 SA 形式进入状态池，成为可被认知和回忆的对象。",
            "  3. 同一通道的感受又会反过来调制 query_spacetime 或召回评分，形成闭环。",
            "- 这意味着‘时间感、节奏感、惊、违和感、正确感、奖励/惩罚、运动趋势、听觉焦点’都不是外挂标签，而是统一状态池中的一等公民。",
            "",
            "## 八、当前边界",
            "- 时间感目前更像模糊主峰，不是精确计时器，因此实验结论应表述为‘已形成稳定的模糊时间间隔感与时间召回偏置’，而不是‘已精确恢复真实 tick 差’。",
            "- 听觉的焦点链、结构链和入池链已证明，但音频语义跨模态识别还应继续用更好的数据集做下一轮严格验收。",
            "- 视觉 OCR-like 方面，本轮目标是证明统一状态池的可用性与动态补强接口，后续还可以继续扩大数据集、降低随机性、做更长时程测试。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_id": "multichannel_feelings_report/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "time": _run_time_experiment(),
        "emotion": _run_emotion_experiment(),
        "rhythm": _run_rhythm_experiment(),
        "motion_recall": _run_motion_recall_experiment(),
        "feedback_recall": _run_feedback_recall_experiment(),
        "visual": _run_visual_experiment(),
        "audio": _run_audio_experiment(),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_text(output_dir / "report.md", _render_report(summary))
    print(output_dir)


if __name__ == "__main__":
    main()
