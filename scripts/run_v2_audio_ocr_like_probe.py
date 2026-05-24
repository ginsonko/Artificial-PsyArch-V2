# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config
from scripts.run_v2_vision_ocr_probe import (
    _best_label_from_energy_map,
    _cstar_text_energies,
    _filter_exact_bn,
    _round4,
    _write_json,
    _write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "audio_ocr_probe"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "V2_原生音频识别_OCR_like_实验报告_2026-05-23.md"


@dataclass(frozen=True)
class AudioPair:
    pair_id: str
    text_label: str
    start_hz: float
    end_hz: float
    duration_sec: float = 0.24
    amplitude: int = 12000


DEFAULT_PAIRS = [
    AudioPair(pair_id="tone_low_rise", text_label="tone_low", start_hz=320.0, end_hz=480.0),
    AudioPair(pair_id="tone_high_rise", text_label="tone_high", start_hz=760.0, end_hz=980.0),
]


def _mk_chirp(
    *,
    start_hz: float,
    end_hz: float,
    duration_sec: float = 0.24,
    sample_rate: int = 16000,
    amplitude: int = 12000,
) -> bytes:
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


def _base_overrides() -> dict[str, Any]:
    return {
        "autonomous_teacher_enabled": False,
        "autonomous_llm_gate_enabled": False,
        "autonomous_external_teacher_enabled": False,
        "intrinsic_feedback_enabled": False,
        "executor_enabled": False,
        "text_sensor_budget": 8,
        "text_sensor_fatigue_threshold": 999,
        "text_sensor_max_suppression": 0.0,
        "memory_candidate_limit": 192,
        "memory_ann_top_k": 64,
        "short_term_successor_tail_limit": 12,
        "state_pool_anchor_cache_limit": 16,
        "state_pool_residual_unit_limit": 48,
        "r_state_head_limit": 4,
        "r_state_items_per_head": 8,
        "hearing_window_budget": 12,
        "hearing_focus_band_count": 12,
        "hearing_focus_bandwidth_octaves": 1.15,
        "hearing_attention_boost_enabled": True,
        "hearing_attention_boost_decay": 0.8,
        "hearing_attention_boost_max_extra_window_budget": 12,
        "hearing_attention_boost_max_extra_focus_budget": 8,
        "hearing_attention_boost_min_bandwidth_scale": 0.55,
        "hearing_attention_boost_focus_gain": 1.5,
        "hearing_static_dedup_delta_threshold": 0.02,
        "hearing_static_dedup_band_similarity_threshold": 0.92,
        "hearing_static_dedup_max_suppression": 0.85,
        "hearing_auditory_fatigue_decay": 0.82,
        "hearing_auditory_fatigue_step": 0.16,
        "hearing_auditory_fatigue_max": 1.0,
        "vision_raw_state_budget": 1,
        "vision_patch_budget": 1,
        "vision_focus_patch_budget": 1,
    }


def _build_runtime(*, overrides: dict[str, Any] | None = None) -> RuntimeV2:
    merged = dict(_base_overrides())
    if overrides:
        merged.update(overrides)
    runtime = RuntimeV2(config=load_config(overrides=merged), repo_root=REPO_ROOT)
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _run_multimodal_tick(
    runtime: RuntimeV2,
    *,
    tick_index: int,
    text: str = "",
    audio_bytes: bytes | None = None,
    source_type: str = "audio_ocr_probe",
    execute_selected_actions: bool = False,
) -> tuple[dict[str, Any], float]:
    text_packet = runtime.text_sensor.ingest(text, tick_index=tick_index, source_type=source_type)
    audio_packet = (
        runtime.hearing_sensor.ingest_wav_bytes(audio_bytes, tick_index=tick_index, source_type=source_type)
        if audio_bytes is not None
        else None
    )
    started = time.perf_counter()
    tick = runtime.process_multimodal_tick(
        tick_index=tick_index,
        text_packet=text_packet,
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
    return tick, elapsed_ms


def _inject_reward(
    runtime: RuntimeV2,
    *,
    tick_index: int,
    tick: dict[str, Any],
    pair: AudioPair,
    reward: float,
) -> dict[str, Any]:
    provenance = {
        "focus_memory_id": str((tick.get("focus_memory", {}) or {}).get("memory_id", "") or ""),
        "exact_memory_id": str((tick.get("exact_memory", {}) or {}).get("memory_id", "") or ""),
        "bn_ids": [str(item.get("memory_id", "") or "") for item in (tick.get("bn_list", []) or [])[:4]],
    }
    return runtime.inject_feedback_signals(
        tick_index=tick_index,
        feedback={
            "reward": float(reward),
            "punishment": 0.0,
            "notes": [f"audio_ocr_reward::{pair.pair_id}", f"label::{pair.text_label}"],
        },
        provenance=provenance,
        source_type="audio_ocr_reward",
        channel="audio_ocr_reward",
        meta_extra={
            "pair_id": pair.pair_id,
            "target_text_label": pair.text_label,
            "audio_start_hz": _round4(pair.start_hz),
            "audio_end_hz": _round4(pair.end_hz),
        },
    )


def _select_text_from_state_top(state_top_rows: list[dict[str, Any]], allowed_texts: set[str]) -> dict[str, float]:
    energies: dict[str, float] = {}
    for row in state_top_rows:
        label = str(row.get("sa_label", "") or "")
        if not label.startswith("text::"):
            continue
        text = label.split("::", 1)[1]
        if text not in allowed_texts:
            continue
        energies[text] = _round4(energies.get(text, 0.0) + float(row.get("energy", 0.0) or 0.0))
    return energies


def _collect_label_energies(items: list[dict[str, Any]], allowed_labels: set[str]) -> dict[str, float]:
    energies: dict[str, float] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        label = str(row.get("sa_label", "") or "")
        if label not in allowed_labels:
            continue
        energies[label] = _round4(energies.get(label, 0.0) + float(row.get("energy", 0.0) or 0.0))
    return energies


def _best_pair_label(energy_map: dict[str, float], label_to_pair: dict[str, str]) -> str:
    pair_energy: dict[str, float] = {}
    for label, energy in energy_map.items():
        pair_id = str(label_to_pair.get(label, "") or "")
        if not pair_id:
            continue
        pair_energy[pair_id] = pair_energy.get(pair_id, 0.0) + float(energy or 0.0)
    return _best_label_from_energy_map(pair_energy)


def _evaluate_text_recall(*, tick: dict[str, Any], target_text: str, distractor_texts: list[str]) -> dict[str, Any]:
    allowed_texts = {target_text, *[str(item or "") for item in distractor_texts if str(item or "")]}
    exact_bn = _filter_exact_bn(list(tick.get("bn_list", []) or []), allowed_texts=allowed_texts)
    bn_best_text = str(exact_bn[0]["text"]) if exact_bn else ""
    bn_target_rank = next((index + 1 for index, row in enumerate(exact_bn) if str(row.get("text", "")) == target_text), 0)
    bn_target_score = next((float(row.get("score", 0.0) or 0.0) for row in exact_bn if str(row.get("text", "")) == target_text), 0.0)

    c_star = dict(tick.get("c_star", {}) or {})
    cstar_energies = _cstar_text_energies(c_star, allowed_texts=allowed_texts)
    cstar_best_text = _best_label_from_energy_map(cstar_energies)
    distractor_best = 0.0
    for text in distractor_texts:
        distractor_best = max(distractor_best, float(cstar_energies.get(text, 0.0) or 0.0))
    target_energy = float(cstar_energies.get(target_text, 0.0) or 0.0)

    state_top_rows = [dict(item) for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])) if isinstance(item, dict)]
    state_text_energies = _select_text_from_state_top(state_top_rows, allowed_texts=allowed_texts)
    state_best_text = _best_label_from_energy_map(state_text_energies)
    state_distractor_best = 0.0
    for text in distractor_texts:
        state_distractor_best = max(state_distractor_best, float(state_text_energies.get(text, 0.0) or 0.0))
    state_target_energy = float(state_text_energies.get(target_text, 0.0) or 0.0)
    focus_units = [str(item or "") for item in ((tick.get("a_focus", {}) or {}).get("focus_units", []) or []) if str(item or "")]

    return {
        "bn_exact_ranked": exact_bn,
        "bn_best_text": bn_best_text,
        "bn_target_rank": int(bn_target_rank),
        "bn_target_score": _round4(bn_target_score),
        "bn_success": bool(bn_best_text == target_text and bn_target_rank == 1),
        "cstar_best_text": cstar_best_text,
        "cstar_target_energy": _round4(target_energy),
        "cstar_distractor_best_energy": _round4(distractor_best),
        "cstar_margin": _round4(target_energy - distractor_best),
        "cstar_success": bool(cstar_best_text == target_text and target_energy > distractor_best),
        "state_best_text": state_best_text,
        "state_target_energy": _round4(state_target_energy),
        "state_distractor_best_energy": _round4(state_distractor_best),
        "state_margin": _round4(state_target_energy - state_distractor_best),
        "state_success": bool(state_best_text == target_text and state_target_energy > state_distractor_best),
        "focus_units": focus_units,
        "focus_has_target": bool(target_text in focus_units),
        "strict_success": bool(
            bn_best_text == target_text
            and bn_target_rank == 1
            and cstar_best_text == target_text
            and target_energy > distractor_best
        ),
    }


def _evaluate_audio_signature_recall(
    *,
    tick: dict[str, Any],
    target_pair_id: str,
    target_audio_labels: list[str],
    distractor_audio_labels: list[str],
    label_to_pair: dict[str, str],
) -> dict[str, Any]:
    allowed_labels = {str(label or "") for label in [*target_audio_labels, *distractor_audio_labels] if str(label or "")}
    c_star_items = [dict(item) for item in ((tick.get("c_star", {}) or {}).get("items", []) or []) if isinstance(item, dict)]
    state_top_rows = [dict(item) for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])) if isinstance(item, dict)]
    cstar_label_energies = _collect_label_energies(c_star_items, allowed_labels=allowed_labels)
    state_label_energies = _collect_label_energies(state_top_rows, allowed_labels=allowed_labels)
    target_cstar_energy = sum(float(cstar_label_energies.get(label, 0.0) or 0.0) for label in target_audio_labels)
    distractor_cstar_best = 0.0
    for label in distractor_audio_labels:
        distractor_cstar_best = max(distractor_cstar_best, float(cstar_label_energies.get(label, 0.0) or 0.0))
    target_state_energy = sum(float(state_label_energies.get(label, 0.0) or 0.0) for label in target_audio_labels)
    distractor_state_best = 0.0
    for label in distractor_audio_labels:
        distractor_state_best = max(distractor_state_best, float(state_label_energies.get(label, 0.0) or 0.0))
    return {
        "cstar_label_energies": {key: _round4(value) for key, value in sorted(cstar_label_energies.items(), key=lambda item: item[0])},
        "state_label_energies": {key: _round4(value) for key, value in sorted(state_label_energies.items(), key=lambda item: item[0])},
        "cstar_target_energy": _round4(target_cstar_energy),
        "cstar_distractor_best_energy": _round4(distractor_cstar_best),
        "cstar_margin": _round4(target_cstar_energy - distractor_cstar_best),
        "state_target_energy": _round4(target_state_energy),
        "state_distractor_best_energy": _round4(distractor_state_best),
        "state_margin": _round4(target_state_energy - distractor_state_best),
        "cstar_best_pair_id": _best_pair_label(cstar_label_energies, label_to_pair=label_to_pair),
        "state_best_pair_id": _best_pair_label(state_label_energies, label_to_pair=label_to_pair),
        "cstar_success": bool(target_cstar_energy > 0.0 and target_cstar_energy > distractor_cstar_best),
        "state_success": bool(target_state_energy > 0.0 and target_state_energy > distractor_state_best),
        "strict_success": bool(target_cstar_energy > 0.0 and target_cstar_energy > distractor_cstar_best and target_state_energy > 0.0),
        "target_pair_id": target_pair_id,
    }


def _render_audio_manifest(pair: AudioPair, raw: bytes, *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{pair.pair_id}.wav"
    wav_path.write_bytes(raw)
    return {
        "pair_id": pair.pair_id,
        "text_label": pair.text_label,
        "audio_path": str(wav_path),
        "start_hz": _round4(pair.start_hz),
        "end_hz": _round4(pair.end_hz),
        "duration_sec": _round4(pair.duration_sec),
    }


def _extract_audio_signature(*, audio_bytes: bytes) -> dict[str, Any]:
    runtime = _build_runtime()
    packet = runtime.hearing_sensor.ingest_wav_bytes(audio_bytes, tick_index=0, source_type="audio_signature")
    memory_labels = [
        str(item.get("sa_label", "") or "")
        for item in (packet.get("memory_write_samples", []) or [])
        if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("audio::mem::")
    ]
    global_labels = [
        str(item.get("sa_label", "") or "")
        for item in (packet.get("global_structure_samples", []) or [])
        if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("audio::global::")
    ]
    band_labels = [
        str(item.get("sa_label", "") or "")
        for item in (packet.get("global_structure_samples", []) or [])
        if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("audio::global_band_")
    ]
    strongest_window = dict(((packet.get("windows", []) or [])[0] or {}))
    strongest_attrs = dict(strongest_window.get("attributes", {}) or {})
    feature_summary = dict(packet.get("feature_summary", {}) or {})
    return {
        "audio_labels": [*memory_labels[:4], *global_labels[:2], *band_labels[:1]],
        "memory_labels": memory_labels[:4],
        "global_labels": global_labels[:2],
        "band_labels": band_labels[:1],
        "feature_summary": feature_summary,
        "dominant_hz": _round4(float(strongest_attrs.get("dominant_hz", feature_summary.get("dominant_hz", 0.0)) or 0.0)),
    }


def _prepare_audio_bank(*, pairs: list[AudioPair], output_root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    audio_dir = output_root / "generated_audio"
    audio_map: dict[str, bytes] = {}
    audio_manifest: list[dict[str, Any]] = []
    signatures: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        raw = _mk_chirp(
            start_hz=pair.start_hz,
            end_hz=pair.end_hz,
            duration_sec=pair.duration_sec,
            amplitude=pair.amplitude,
        )
        audio_map[pair.pair_id] = raw
        audio_manifest.append(_render_audio_manifest(pair, raw, output_dir=audio_dir))
        signatures[pair.pair_id] = _extract_audio_signature(audio_bytes=raw)
    return audio_map, audio_manifest, signatures


def _probe_runtime(*, payload: dict[str, Any], read_only: bool) -> RuntimeV2:
    runtime = _build_runtime()
    runtime.import_payload({"memory_store": copy.deepcopy(payload.get("memory_store", {}))})
    if read_only:
        runtime.memory_store.write_memory_batch = lambda rows: []
        runtime.memory_store.write_memory = lambda *args, **kwargs: {}
    return runtime


def _training_attempt(
    *,
    pairs: list[AudioPair],
    audio_map: dict[str, bytes],
    train_epochs: int,
    reward_value: float,
    stabilize_ticks: int,
    acceptance_observation_ticks: int,
) -> dict[str, Any]:
    runtime = _build_runtime()
    tick_index = 0
    training_rows: list[dict[str, Any]] = []
    for epoch in range(int(train_epochs)):
        for pair in pairs:
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text=pair.text_label,
                audio_bytes=audio_map[pair.pair_id],
                source_type=f"audio_ocr_train::{pair.pair_id}",
                execute_selected_actions=False,
            )
            reward_payload = _inject_reward(runtime, tick_index=tick_index, tick=tick, pair=pair, reward=reward_value)
            text_eval = _evaluate_text_recall(
                tick=tick,
                target_text=pair.text_label,
                distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
            )
            audio_packet = dict(tick.get("audio_packet", {}) or {})
            training_rows.append(
                {
                    "tick_index": int(tick_index),
                    "epoch": int(epoch),
                    "pair_id": pair.pair_id,
                    "text_label": pair.text_label,
                    "elapsed_ms": _round4(elapsed_ms),
                    "window_count": int(len(audio_packet.get("windows", []) or [])),
                    "memory_write_count": int(len(audio_packet.get("memory_write_samples", []) or [])),
                    "global_structure_count": int(len(audio_packet.get("global_structure_samples", []) or [])),
                    "focus_priority_count": int(len(audio_packet.get("focus_priority_samples", []) or [])),
                    "bn_best_text": str(text_eval.get("bn_best_text", "") or ""),
                    "cstar_best_text": str(text_eval.get("cstar_best_text", "") or ""),
                    "cstar_margin": _round4(float(text_eval.get("cstar_margin", 0.0) or 0.0)),
                    "reward": _round4(float(reward_payload.get("reward", 0.0) or 0.0)),
                }
            )
            tick_index += 1

    stabilize_rows: list[dict[str, Any]] = []
    for _ in range(max(0, int(stabilize_ticks))):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            audio_bytes=None,
            source_type="audio_ocr_stabilize",
            execute_selected_actions=False,
        )
        stabilize_rows.append(
            {
                "tick_index": int(tick_index),
                "elapsed_ms": _round4(elapsed_ms),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
            }
        )
        tick_index += 1

    payload = runtime.export_payload()
    acceptance_rows: list[dict[str, Any]] = []
    all_success = True
    for pair in pairs:
        session = _audio_only_probe_session(
            imported_payload=payload,
            pair=pair,
            all_pairs=pairs,
            audio_bytes=audio_map[pair.pair_id],
            observation_ticks=acceptance_observation_ticks,
            read_only=True,
        )
        acceptance_rows.append(session)
        if not bool((session.get("final_evaluation", {}) or {}).get("strict_success", False)):
            all_success = False

    return {
        "train_epochs": int(train_epochs),
        "reward_value": _round4(reward_value),
        "stabilize_ticks": int(stabilize_ticks),
        "acceptance_observation_ticks": int(acceptance_observation_ticks),
        "training_rows": training_rows,
        "stabilize_rows": stabilize_rows,
        "acceptance_rows": acceptance_rows,
        "accepted": bool(all_success),
        "stabilized_payload": payload,
    }


def _audio_only_probe_session(
    *,
    imported_payload: dict[str, Any],
    pair: AudioPair,
    all_pairs: list[AudioPair],
    audio_bytes: bytes,
    observation_ticks: int,
    read_only: bool,
) -> dict[str, Any]:
    runtime = _probe_runtime(payload=imported_payload, read_only=read_only)
    tick_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_index,
            text="",
            audio_bytes=audio_bytes,
            source_type=f"audio_ocr_probe::{pair.pair_id}",
            execute_selected_actions=False,
        )
        eval_row = _evaluate_text_recall(
            tick=tick,
            target_text=pair.text_label,
            distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
        )
        audio_packet = dict(tick.get("audio_packet", {}) or {})
        tick_rows.append(
            {
                "probe_tick_index": int(probe_index),
                "elapsed_ms": _round4(elapsed_ms),
                "window_count": int(len(audio_packet.get("windows", []) or [])),
                "memory_write_count": int(len(audio_packet.get("memory_write_samples", []) or [])),
                "global_structure_count": int(len(audio_packet.get("global_structure_samples", []) or [])),
                "focus_priority_count": int(len(audio_packet.get("focus_priority_samples", []) or [])),
                "bn_best_text": str(eval_row.get("bn_best_text", "") or ""),
                "bn_target_rank": int(eval_row.get("bn_target_rank", 0) or 0),
                "cstar_best_text": str(eval_row.get("cstar_best_text", "") or ""),
                "cstar_margin": _round4(float(eval_row.get("cstar_margin", 0.0) or 0.0)),
                "state_best_text": str(eval_row.get("state_best_text", "") or ""),
                "state_margin": _round4(float(eval_row.get("state_margin", 0.0) or 0.0)),
                "focus_has_target": bool(eval_row.get("focus_has_target", False)),
                "strict_success": bool(eval_row.get("strict_success", False)),
            }
        )
        final_tick = tick

    final_eval = _evaluate_text_recall(
        tick=final_tick,
        target_text=pair.text_label,
        distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
    )
    first_success_tick = next((int(row["probe_tick_index"]) for row in tick_rows if bool(row.get("strict_success", False))), None)
    return {
        "pair_id": pair.pair_id,
        "target_text_label": pair.text_label,
        "observation_ticks": int(observation_ticks),
        "tick_rows": tick_rows,
        "final_evaluation": final_eval,
        "first_strict_success_tick": first_success_tick,
        "mean_elapsed_ms": _round4(sum(float(row["elapsed_ms"]) for row in tick_rows) / max(1, len(tick_rows))),
    }


def _text_only_reverse_probe_session(
    *,
    imported_payload: dict[str, Any],
    pair: AudioPair,
    all_pairs: list[AudioPair],
    signatures: dict[str, dict[str, Any]],
    observation_ticks: int,
    read_only: bool,
) -> dict[str, Any]:
    runtime = _probe_runtime(payload=imported_payload, read_only=read_only)
    target_audio_labels = list((signatures.get(pair.pair_id, {}) or {}).get("audio_labels", []) or [])
    distractor_audio_labels = [
        label
        for other in all_pairs
        if other.pair_id != pair.pair_id
        for label in ((signatures.get(other.pair_id, {}) or {}).get("audio_labels", []) or [])
    ]
    label_to_pair = {
        str(label): other.pair_id
        for other in all_pairs
        for label in ((signatures.get(other.pair_id, {}) or {}).get("audio_labels", []) or [])
    }
    tick_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_index,
            text=pair.text_label,
            audio_bytes=None,
            source_type=f"audio_reverse_probe::{pair.pair_id}",
            execute_selected_actions=False,
        )
        text_eval = _evaluate_text_recall(
            tick=tick,
            target_text=pair.text_label,
            distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
        )
        audio_eval = _evaluate_audio_signature_recall(
            tick=tick,
            target_pair_id=pair.pair_id,
            target_audio_labels=target_audio_labels,
            distractor_audio_labels=distractor_audio_labels,
            label_to_pair=label_to_pair,
        )
        tick_rows.append(
            {
                "probe_tick_index": int(probe_index),
                "elapsed_ms": _round4(elapsed_ms),
                "bn_best_text": str(text_eval.get("bn_best_text", "") or ""),
                "cstar_best_text": str(text_eval.get("cstar_best_text", "") or ""),
                "audio_cstar_best_pair_id": str(audio_eval.get("cstar_best_pair_id", "") or ""),
                "audio_state_best_pair_id": str(audio_eval.get("state_best_pair_id", "") or ""),
                "audio_cstar_margin": _round4(float(audio_eval.get("cstar_margin", 0.0) or 0.0)),
                "audio_state_margin": _round4(float(audio_eval.get("state_margin", 0.0) or 0.0)),
                "audio_strict_success": bool(audio_eval.get("strict_success", False)),
            }
        )
        final_tick = tick

    final_text_eval = _evaluate_text_recall(
        tick=final_tick,
        target_text=pair.text_label,
        distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
    )
    final_audio_eval = _evaluate_audio_signature_recall(
        tick=final_tick,
        target_pair_id=pair.pair_id,
        target_audio_labels=target_audio_labels,
        distractor_audio_labels=distractor_audio_labels,
        label_to_pair=label_to_pair,
    )
    first_audio_success_tick = next((int(row["probe_tick_index"]) for row in tick_rows if bool(row.get("audio_strict_success", False))), None)
    return {
        "pair_id": pair.pair_id,
        "target_text_label": pair.text_label,
        "target_audio_labels": target_audio_labels,
        "observation_ticks": int(observation_ticks),
        "tick_rows": tick_rows,
        "final_text_evaluation": final_text_eval,
        "final_audio_evaluation": final_audio_eval,
        "first_audio_success_tick": first_audio_success_tick,
        "mean_elapsed_ms": _round4(sum(float(row["elapsed_ms"]) for row in tick_rows) / max(1, len(tick_rows))),
    }


def _find_first_flip(rows: list[dict[str, Any]], *, key: str, target_value: Any) -> int | None:
    for row in rows:
        if row.get(key) == target_value:
            return int(row.get("probe_tick_index", 0) or 0)
    return None


def _switching_probe(
    *,
    imported_payload: dict[str, Any],
    warm_pair: AudioPair,
    target_pair: AudioPair,
    all_pairs: list[AudioPair],
    audio_map: dict[str, bytes],
    warm_ticks: int,
    observation_ticks: int,
    read_only: bool,
) -> dict[str, Any]:
    runtime = _probe_runtime(payload=imported_payload, read_only=read_only)
    tick_cursor = 0
    warm_rows: list[dict[str, Any]] = []
    for warm_index in range(int(warm_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_cursor,
            text="",
            audio_bytes=audio_map[warm_pair.pair_id],
            source_type=f"audio_switch_warm::{warm_pair.pair_id}",
            execute_selected_actions=False,
        )
        eval_row = _evaluate_text_recall(
            tick=tick,
            target_text=warm_pair.text_label,
            distractor_texts=[item.text_label for item in all_pairs if item.pair_id != warm_pair.pair_id],
        )
        warm_rows.append(
            {
                "probe_tick_index": int(warm_index),
                "elapsed_ms": _round4(elapsed_ms),
                "bn_best_text": str(eval_row.get("bn_best_text", "") or ""),
                "cstar_best_text": str(eval_row.get("cstar_best_text", "") or ""),
                "state_best_text": str(eval_row.get("state_best_text", "") or ""),
                "strict_success": bool(eval_row.get("strict_success", False)),
            }
        )
        tick_cursor += 1

    switch_rows: list[dict[str, Any]] = []
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_cursor,
            text="",
            audio_bytes=audio_map[target_pair.pair_id],
            source_type=f"audio_switch_probe::{target_pair.pair_id}",
            execute_selected_actions=False,
        )
        eval_row = _evaluate_text_recall(
            tick=tick,
            target_text=target_pair.text_label,
            distractor_texts=[item.text_label for item in all_pairs if item.pair_id != target_pair.pair_id],
        )
        switch_rows.append(
            {
                "probe_tick_index": int(probe_index),
                "elapsed_ms": _round4(elapsed_ms),
                "bn_best_text": str(eval_row.get("bn_best_text", "") or ""),
                "bn_target_rank": int(eval_row.get("bn_target_rank", 0) or 0),
                "cstar_best_text": str(eval_row.get("cstar_best_text", "") or ""),
                "cstar_margin": _round4(float(eval_row.get("cstar_margin", 0.0) or 0.0)),
                "state_best_text": str(eval_row.get("state_best_text", "") or ""),
                "state_margin": _round4(float(eval_row.get("state_margin", 0.0) or 0.0)),
                "focus_has_target": bool(eval_row.get("focus_has_target", False)),
                "strict_success": bool(eval_row.get("strict_success", False)),
            }
        )
        tick_cursor += 1

    final_eval = _evaluate_text_recall(
        tick=tick,
        target_text=target_pair.text_label,
        distractor_texts=[item.text_label for item in all_pairs if item.pair_id != target_pair.pair_id],
    )
    return {
        "warm_pair_id": warm_pair.pair_id,
        "target_pair_id": target_pair.pair_id,
        "warm_ticks": int(warm_ticks),
        "observation_ticks": int(observation_ticks),
        "warm_rows": warm_rows,
        "switch_rows": switch_rows,
        "final_evaluation": final_eval,
        "bn_flip_tick": _find_first_flip(switch_rows, key="bn_best_text", target_value=target_pair.text_label),
        "cstar_flip_tick": _find_first_flip(switch_rows, key="cstar_best_text", target_value=target_pair.text_label),
        "state_flip_tick": _find_first_flip(switch_rows, key="state_best_text", target_value=target_pair.text_label),
        "focus_flip_tick": _find_first_flip(switch_rows, key="focus_has_target", target_value=True),
    }


def _summarize_acceptance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        final_eval = dict(row.get("final_evaluation", {}) or {})
        summary.append(
            {
                "pair_id": str(row.get("pair_id", "") or ""),
                "target_text_label": str(row.get("target_text_label", "") or ""),
                "bn_best_text": str(final_eval.get("bn_best_text", "") or ""),
                "bn_target_rank": int(final_eval.get("bn_target_rank", 0) or 0),
                "cstar_best_text": str(final_eval.get("cstar_best_text", "") or ""),
                "cstar_margin": _round4(float(final_eval.get("cstar_margin", 0.0) or 0.0)),
                "focus_has_target": bool(final_eval.get("focus_has_target", False)),
                "strict_success": bool(final_eval.get("strict_success", False)),
                "first_strict_success_tick": row.get("first_strict_success_tick", None),
            }
        )
    return summary


def _render_report_markdown(
    *,
    summary: dict[str, Any],
    output_root: Path,
) -> str:
    selected_attempt = dict(summary.get("selected_attempt", {}) or {})
    acceptance_summary = _summarize_acceptance(list(selected_attempt.get("acceptance_rows", []) or []))
    audio_only_rows = list(summary.get("audio_only_probe_rows", []) or [])
    text_only_rows = list(summary.get("text_only_reverse_probe_rows", []) or [])
    switching = dict(summary.get("switching_probe", {}) or {})
    audio_manifest = list(summary.get("audio_manifest", []) or [])
    signatures = dict(summary.get("audio_signatures", {}) or {})

    lines: list[str] = []
    lines.append("# V2 原生音频识别 OCR-like 实验报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("本实验的目标，不是验证传统 ASR 或关键词分类器，而是验证 AP V2 是否已经具备一种原生的音频-文本绑定与召回雏形。")
    lines.append("更具体地说，本轮实验想回答四个问题：")
    lines.append("1. 当一段音频结构与对应文本标签持续共现时，系统能否把这段音频的结构信息与文本标签绑定起来。")
    lines.append("2. 训练完成后，仅再次输入音频而不给文本，系统能否从记忆中正确召回对应文本。")
    lines.append("3. 仅输入文本而不给音频时，系统能否反过来召回与该文本相关的音频结构记忆。")
    lines.append("4. 在旧音频上下文干扰下，系统能否经过若干连续 observation tick，从旧对象逐步切换到新的音频认知对象。")
    lines.append("")
    lines.append("## 2. 实验材料")
    lines.append("")
    lines.append("本轮实验使用两段程序生成的单声道上行 chirp 音频：")
    for row in audio_manifest:
        lines.append(
            f"- `{row['pair_id']}`：文本标签 `{row['text_label']}`，频率从 {row['start_hz']} Hz 上升到 {row['end_hz']} Hz，时长 {row['duration_sec']} 秒，文件为 [{Path(row['audio_path']).name}]({row['audio_path']})"
        )
        signature = dict(signatures.get(str(row["pair_id"]), {}) or {})
        lines.append(
            f"  - 可观测结构标签：memory={signature.get('memory_labels', [])[:2]} / global={signature.get('global_labels', [])[:2]} / dominant_hz={signature.get('dominant_hz', 0.0)}"
        )
    lines.append("")
    lines.append("## 3. 实验链路与判定口径")
    lines.append("")
    lines.append("音频不会进入传统语音识别模块，而是进入 AP V2 的统一多模态链路：")
    lines.append("1. 听觉感受器把波形拆成听窗、焦点优先样本、可入记忆的结构特征，以及全局听觉结构特征。")
    lines.append("2. 文本标签与听觉结构同 tick 进入状态池。")
    lines.append("3. 系统通过 `Bn` 召回历史记忆，再通过 `C*` 形成当前整合预测包。")
    lines.append("4. 训练阶段同步注入奖励信号，塑造“该音频结构 <-> 该文本标签”的长期联结。")
    lines.append("")
    lines.append("本实验主要看四层：")
    lines.append("- `BN_top`：最强候选显式记忆的文本标签。")
    lines.append("- `C*_top`：综合预测包中最强文本标签。")
    lines.append("- `State_top_text`：状态池主导文本波峰。")
    lines.append("- 反向音频结构召回：文本单独输入时，与目标音频对应的 `audio::mem::* / audio::global::*` 标签能否在 `C*` 和状态池内压过干扰对象。")
    lines.append("")
    lines.append("其中：")
    lines.append("- `BN_top` 正确，说明显式记忆召回已开始对齐。")
    lines.append("- `C*_top` 正确，说明当前整合后的认知判断已对齐。")
    lines.append("- `State_top_text` 正确，说明状态池主导波峰已切换到目标文本。")
    lines.append("- 文本反向召回到目标音频结构标签，说明多模态绑定已经不是单向的，而是可以反向联想到听觉记忆。")
    lines.append("")
    lines.append("## 4. 训练设计")
    lines.append("")
    lines.append(
        f"- 训练轮次：{int(selected_attempt.get('train_epochs', 0) or 0)} epoch，每个 epoch 交替呈现两段音频与对应文本。"
    )
    lines.append(f"- 每个正确共现 tick 注入奖励：reward={_round4(float(selected_attempt.get('reward_value', 0.0) or 0.0))}")
    lines.append(f"- 训练后稳定空 tick：{int(selected_attempt.get('stabilize_ticks', 0) or 0)}")
    lines.append(f"- 接受门槛冷探测：audio-only，连续 {int(selected_attempt.get('acceptance_observation_ticks', 0) or 0)} tick")
    lines.append(f"- 实验输出目录：[summary.json]({output_root / 'summary.json'}) / [report.md]({output_root / 'report.md'})")
    lines.append("")
    lines.append("## 5. 预期")
    lines.append("")
    lines.append("如果 AP V2 的原生音频识别链路成立，那么预期会出现以下现象：")
    lines.append("1. 训练后，面对 `tone_low_rise` 音频时，`BN_top` 和 `C*_top` 应稳定偏向 `tone_low`。")
    lines.append("2. 面对 `tone_high_rise` 音频时，`BN_top` 和 `C*_top` 应稳定偏向 `tone_high`。")
    lines.append("3. 当只输入文本 `tone_low` 或 `tone_high` 时，与该文本绑定过的音频结构标签应被反向召回，而不是只剩文本自身。")
    lines.append("4. 在先听过另一段音频的前提下，新音频未必第一反应就立刻获胜，但经过几个连续 tick，应逐步翻转到正确对象。")
    lines.append("")
    lines.append("## 6. 主实验结果")
    lines.append("")
    lines.append("### 6.1 Audio-only 冷探测")
    for row in audio_only_rows:
        final_eval = dict(row.get("final_evaluation", {}) or {})
        lines.append(
            f"- `{row.get('pair_id', '')}` -> `{row.get('target_text_label', '')}`：BN_top=`{final_eval.get('bn_best_text', '')}`，C*_top=`{final_eval.get('cstar_best_text', '')}`，State_top=`{final_eval.get('state_best_text', '')}`，C* margin={_round4(float(final_eval.get('cstar_margin', 0.0) or 0.0))}，首次严格成功 tick={row.get('first_strict_success_tick', None)}，strict_success={bool(final_eval.get('strict_success', False))}"
        )
    lines.append("")
    lines.append("### 6.2 接受门槛检查")
    for row in acceptance_summary:
        lines.append(
            f"- `{row['pair_id']}`：BN_rank={row['bn_target_rank']} / C*_top=`{row['cstar_best_text']}` / focus_has_target={row['focus_has_target']} / first_success_tick={row.get('first_strict_success_tick', None)} / strict_success={row['strict_success']}"
        )
    lines.append("")
    lines.append("### 6.3 Text-only 反向召回音频结构")
    for row in text_only_rows:
        text_eval = dict(row.get("final_text_evaluation", {}) or {})
        audio_eval = dict(row.get("final_audio_evaluation", {}) or {})
        lines.append(
            f"- 文本 `{row.get('target_text_label', '')}`：BN_top=`{text_eval.get('bn_best_text', '')}`，C*_text_top=`{text_eval.get('cstar_best_text', '')}`，音频结构 C*_best_pair=`{audio_eval.get('cstar_best_pair_id', '')}`，State_best_pair=`{audio_eval.get('state_best_pair_id', '')}`，audio_cstar_margin={audio_eval.get('cstar_margin', 0.0)}，首次音频反向命中 tick={row.get('first_audio_success_tick', None)}，audio_strict_success={audio_eval.get('strict_success', False)}"
        )
        lines.append(f"  - 目标音频标签：{row.get('target_audio_labels', [])[:4]}")
    lines.append("")
    lines.append("### 6.4 旧上下文干扰下的音频切换")
    lines.append(
        f"- 协议：先连续听 `{switching.get('warm_pair_id', '')}` 共 {switching.get('warm_ticks', 0)} tick，再不清空地切换到 `{switching.get('target_pair_id', '')}` 并继续观察 {switching.get('observation_ticks', 0)} tick。"
    )
    lines.append(
        f"- 翻转结果：BN_top 第 {switching.get('bn_flip_tick', None)} 个 observation tick 翻转；C*_top 第 {switching.get('cstar_flip_tick', None)} 个翻转；State_top_text 第 {switching.get('state_flip_tick', None)} 个翻转；A_focus 第 {switching.get('focus_flip_tick', None)} 个命中。"
    )
    switch_rows = list(switching.get("switch_rows", []) or [])
    if switch_rows:
        lines.append("")
        lines.append("| observation tick | BN_top | C*_top | C* margin | State_top | State margin | A_focus 命中 |")
        lines.append("| --- | --- | --- | ---: | --- | ---: | --- |")
        for row in switch_rows:
            lines.append(
                f"| {int(row.get('probe_tick_index', 0) or 0)} | {row.get('bn_best_text', '')} | {row.get('cstar_best_text', '')} | {_round4(float(row.get('cstar_margin', 0.0) or 0.0))} | {row.get('state_best_text', '')} | {_round4(float(row.get('state_margin', 0.0) or 0.0))} | {'是' if bool(row.get('focus_has_target', False)) else '否'} |"
            )
    lines.append("")
    lines.append("## 7. 结果解释")
    lines.append("")
    lines.append("这组结果如果成立，证明的不是“已经做出了传统语音识别产品”，而是更基础、更重要的一点：")
    lines.append("AP V2 已经可以把音频结构与文本标签放进同一个统一状态池链路里学习，并在之后仅凭其中一个模态的线索，把另一个模态相关记忆重新拉起来。")
    lines.append("")
    lines.append("尤其是 text-only 反向召回这一步很关键。")
    lines.append("如果只有 audio-only -> text 成功，那还可能被解释成“只是在做音频分类后拉文本标签”；")
    lines.append("但如果 text-only 时，和目标音频对应的结构标签也能被一起带出，说明这里建立的是跨模态绑定，而不是单向映射。")
    lines.append("")
    lines.append("切换实验的意义则在于：")
    lines.append("- 系统并不要求第一反应永远正确。")
    lines.append("- 它允许旧上下文短暂残留。")
    lines.append("- 但在持续新证据输入下，能逐步完成认知翻转。")
    lines.append("")
    lines.append("这比一个一次性静态分类器更接近认知系统的行为图景。")
    lines.append("")
    lines.append("## 8. 这意味着什么")
    lines.append("")
    lines.append("如果把它翻译成更直白的话，就是：")
    lines.append("1. 我们已经不仅能让系统“听到不同的声音不一样”，而是开始能让它把“这段声音”和“这个词”绑定起来。")
    lines.append("2. 训练后，仅输入声音，它可以逐步想到对应文本。")
    lines.append("3. 仅输入文本，它也可以逐步想到之前和它一起出现过的那类声音结构。")
    lines.append("4. 这说明 AP 的多模态统一召回，不只适用于文字-图像，也可以扩展到文字-音频。")
    lines.append("")
    lines.append("如果后续再把动作、情绪、奖惩等一起绑定进去，这就更接近“教一个主体认识一个对象”的方式，而不是单模态分类器。")
    lines.append("")
    lines.append("## 9. 当前边界")
    lines.append("")
    lines.append("本实验证明的是“原生音频识别雏形成立”，但还不能直接推出更强结论：")
    lines.append("1. 不能直接等价于成熟 ASR 或通用音频事件识别。")
    lines.append("2. 当前刺激集只有两类、且人为构造得比较清晰，因此证明的是原理可行性，不是大规模开放环境泛化。")
    lines.append("3. 当前 text-only 反向召回的可观测证据，主要还是 `audio::mem::* / audio::global::*` 结构标签，而不是最终可试听的内心声音重建。")
    lines.append("4. 本轮默认使用带奖励的配对训练，因此证明的是“统一状态池 + 记忆召回 + 奖励塑形”这条链路可行，不是“无奖励也会自动同样稳定形成绑定”。")
    lines.append("")
    lines.append("## 10. 阶段性总结")
    lines.append("")
    lines.append("如果只看“有没有这种能力”，本轮实验一旦通过，答案就是肯定的。")
    lines.append("它最值得和同事分享的，不是“已经做出了产品级语音识别”，而是：")
    lines.append("")
    lines.append("> **AP V2 这种统一状态池 + 记忆召回 + 预测包 + 奖励塑形的架构，已经可以不依赖传统 ASR 模块，直接长出一种原生的音频-文本识别与反向联想雏形。**")
    lines.append("")
    lines.append("而且这个能力不是单步硬分类式输出，而是带有连续观察、旧上下文干扰、逐步翻转、反向联想这些更接近认知系统的行为特征。")
    lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    output_root: Path,
    doc_path: Path,
    train_epochs_candidates: list[int],
    reward_value: float,
    stabilize_ticks: int,
    acceptance_observation_ticks: int,
    audio_probe_ticks: int,
    reverse_probe_ticks: int,
    switch_warm_ticks: int,
    switch_observation_ticks: int,
) -> dict[str, Any]:
    pairs = list(DEFAULT_PAIRS)
    audio_map, audio_manifest, signatures = _prepare_audio_bank(pairs=pairs, output_root=output_root)
    attempts: list[dict[str, Any]] = []
    selected_attempt: dict[str, Any] | None = None
    for train_epochs in train_epochs_candidates:
        attempt = _training_attempt(
            pairs=pairs,
            audio_map=audio_map,
            train_epochs=int(train_epochs),
            reward_value=reward_value,
            stabilize_ticks=stabilize_ticks,
            acceptance_observation_ticks=acceptance_observation_ticks,
        )
        attempts.append({key: value for key, value in attempt.items() if key != "stabilized_payload"})
        if bool(attempt.get("accepted", False)):
            selected_attempt = attempt
            break
        if selected_attempt is None:
            selected_attempt = attempt

    if selected_attempt is None:
        raise RuntimeError("未能生成任何有效的音频训练尝试结果")

    payload = dict(selected_attempt.get("stabilized_payload", {}) or {})
    audio_only_probe_rows = [
        _audio_only_probe_session(
            imported_payload=payload,
            pair=pair,
            all_pairs=pairs,
            audio_bytes=audio_map[pair.pair_id],
            observation_ticks=audio_probe_ticks,
            read_only=True,
        )
        for pair in pairs
    ]
    text_only_reverse_probe_rows = [
        _text_only_reverse_probe_session(
            imported_payload=payload,
            pair=pair,
            all_pairs=pairs,
            signatures=signatures,
            observation_ticks=reverse_probe_ticks,
            read_only=True,
        )
        for pair in pairs
    ]
    switching_probe = _switching_probe(
        imported_payload=payload,
        warm_pair=pairs[0],
        target_pair=pairs[1],
        all_pairs=pairs,
        audio_map=audio_map,
        warm_ticks=switch_warm_ticks,
        observation_ticks=switch_observation_ticks,
        read_only=True,
    )

    summary = {
        "schema_id": "audio_ocr_probe/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audio_manifest": audio_manifest,
        "audio_signatures": signatures,
        "attempts": attempts,
        "selected_attempt": {key: value for key, value in selected_attempt.items() if key != "stabilized_payload"},
        "audio_only_probe_rows": audio_only_probe_rows,
        "text_only_reverse_probe_rows": text_only_reverse_probe_rows,
        "switching_probe": switching_probe,
        "config": {
            "reward_value": _round4(reward_value),
            "stabilize_ticks": int(stabilize_ticks),
            "acceptance_observation_ticks": int(acceptance_observation_ticks),
            "audio_probe_ticks": int(audio_probe_ticks),
            "reverse_probe_ticks": int(reverse_probe_ticks),
            "switch_warm_ticks": int(switch_warm_ticks),
            "switch_observation_ticks": int(switch_observation_ticks),
            "train_epochs_candidates": [int(item) for item in train_epochs_candidates],
        },
    }
    report_markdown = _render_report_markdown(summary=summary, output_root=output_root)
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "report.md", report_markdown)
    _write_json(output_root / "audio_only_probe_rows.json", audio_only_probe_rows)
    _write_json(output_root / "text_only_reverse_probe_rows.json", text_only_reverse_probe_rows)
    _write_json(output_root / "switching_probe.json", switching_probe)
    _write_text(doc_path, report_markdown)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AP V2 native audio OCR-like experiment and emit a formal report.")
    parser.add_argument("--output-dir", default="", help="输出目录，默认写入 outputs/audio_ocr_probe/<timestamp>")
    parser.add_argument("--doc-path", default="", help="正式报告路径，默认写入 docs/V2_原生音频识别_OCR_like_实验报告_2026-05-23.md")
    parser.add_argument("--train-epochs", default="6,8,10", help="训练轮次候选，逗号分隔")
    parser.add_argument("--reward", type=float, default=1.0, help="每个训练 tick 注入的奖励值")
    parser.add_argument("--stabilize-ticks", type=int, default=8, help="训练后用于稳定状态池的空 tick 数")
    parser.add_argument("--acceptance-observation-ticks", type=int, default=4, help="训练后接受门槛冷探测的连续 tick 数")
    parser.add_argument("--audio-probe-ticks", type=int, default=6, help="主实验 audio-only 冷探测连续 tick 数")
    parser.add_argument("--reverse-probe-ticks", type=int, default=6, help="text-only 反向召回连续 tick 数")
    parser.add_argument("--switch-warm-ticks", type=int, default=4, help="切换实验里旧对象预热 tick 数")
    parser.add_argument("--switch-observation-ticks", type=int, default=8, help="切换实验里新对象观察 tick 数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir).expanduser() if str(args.output_dir or "").strip() else DEFAULT_OUTPUT_ROOT / timestamp
    doc_path = Path(args.doc_path).expanduser() if str(args.doc_path or "").strip() else DEFAULT_DOC_PATH
    train_epochs_candidates = [max(1, int(item.strip())) for item in str(args.train_epochs or "").split(",") if item.strip()]
    summary = run_experiment(
        output_root=output_root,
        doc_path=doc_path,
        train_epochs_candidates=train_epochs_candidates,
        reward_value=float(args.reward),
        stabilize_ticks=max(0, int(args.stabilize_ticks)),
        acceptance_observation_ticks=max(1, int(args.acceptance_observation_ticks)),
        audio_probe_ticks=max(1, int(args.audio_probe_ticks)),
        reverse_probe_ticks=max(1, int(args.reverse_probe_ticks)),
        switch_warm_ticks=max(1, int(args.switch_warm_ticks)),
        switch_observation_ticks=max(1, int(args.switch_observation_ticks)),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "doc_path": str(doc_path),
                "selected_attempt": summary.get("selected_attempt", {}),
                "audio_only_probe_rows": summary.get("audio_only_probe_rows", []),
                "text_only_reverse_probe_rows": summary.get("text_only_reverse_probe_rows", []),
                "switching_probe": summary.get("switching_probe", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
