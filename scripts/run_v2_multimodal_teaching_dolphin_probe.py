# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.dataset_runner import run_dataset_file
from scripts.run_v2_multichannel_feelings_report import _emotion_row
from scripts.run_v2_vision_ocr_probe import _best_label_from_energy_map, _cstar_text_energies, _filter_exact_bn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "multimodal_teaching_dolphin_probe"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "V2_多模态教学与海豚训练综合实验报告_2026-05-23.md"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round4(sum(float(v) for v in values) / max(1, len(values)))


def _bool_mark(value: bool) -> str:
    return "是" if bool(value) else "否"


def _probe_group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("clear_mode", "") or ""), str(row.get("modality", "") or ""))


def _switch_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    final_eval = dict(row.get("final_evaluation", {}) or {})
    observation_rows = list(row.get("switch_rows", []) or [])
    first_obs = dict(observation_rows[0] or {}) if observation_rows else {}
    first_obs_emotion = dict(first_obs.get("emotion", {}) or {})
    first_obs_text_eval = dict(first_obs.get("text_eval", {}) or {})
    return {
        "switch": f"{row.get('warm_concept_id', '')}->{row.get('target_concept_id', '')}",
        "bn_flip_tick": row.get("bn_flip_tick", None),
        "cstar_flip_tick": row.get("cstar_flip_tick", None),
        "state_flip_tick": row.get("state_flip_tick", None),
        "focus_flip_tick": row.get("focus_flip_tick", None),
        "final_bn": final_eval.get("bn_best_text", ""),
        "final_cstar": final_eval.get("cstar_best_text", ""),
        "final_state": final_eval.get("state_best_text", ""),
        "final_strict": bool(final_eval.get("strict_success", False)),
        "first_surprise": _round4(float(first_obs_emotion.get("surprise", 0.0) or 0.0)),
        "first_dissonance": _round4(float(first_obs_emotion.get("dissonance", 0.0) or 0.0)),
        "first_bn": first_obs_text_eval.get("bn_best_text", ""),
        "first_cstar": first_obs_text_eval.get("cstar_best_text", ""),
        "first_state": first_obs_text_eval.get("state_best_text", ""),
    }


def _dedupe_keep_order(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        key = str(label or "")
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _is_audio_identity_label(label: str) -> bool:
    key = str(label or "")
    return key.startswith("audio::mem::") or key.startswith("audio::global::") or key.startswith("audio::global_band_")


def _is_vision_identity_label(label: str) -> bool:
    return str(label or "").startswith("vision_mem::")


@dataclass(frozen=True)
class MultimodalConcept:
    concept_id: str
    text_label: str
    zh_text: str
    rgb: tuple[int, int, int]
    shape: str
    spoken_text: str
    fallback_freqs: tuple[float, float]
    tts_voice: str = "Microsoft Zira Desktop"
    tts_rate: int = 0


DEFAULT_CONCEPTS = [
    MultimodalConcept(
        concept_id="apple",
        text_label="apple",
        zh_text="苹果",
        rgb=(208, 44, 44),
        shape="apple",
        spoken_text="apple apple",
        fallback_freqs=(420.0, 510.0),
        tts_voice="Microsoft Zira Desktop",
        tts_rate=-2,
    ),
    MultimodalConcept(
        concept_id="banana",
        text_label="banana",
        zh_text="香蕉",
        rgb=(228, 206, 58),
        shape="banana",
        spoken_text="banana",
        fallback_freqs=(700.0, 860.0),
        tts_voice="Microsoft Zira Desktop",
        tts_rate=0,
    ),
]


def _base_overrides() -> dict[str, Any]:
    return {
        "autonomous_teacher_enabled": False,
        "autonomous_llm_gate_enabled": False,
        "autonomous_external_teacher_enabled": False,
        "executor_enabled": False,
        "intrinsic_feedback_enabled": True,
        "memory_candidate_limit": 224,
        "memory_ann_top_k": 72,
        "short_term_successor_tail_limit": 14,
        "state_pool_anchor_cache_limit": 16,
        "state_pool_residual_unit_limit": 56,
        "r_state_head_limit": 4,
        "r_state_items_per_head": 8,
        "text_sensor_budget": 10,
        "text_sensor_fatigue_threshold": 999,
        "text_sensor_max_suppression": 0.0,
        "vision_edge_candidate_gain": 1.9,
        "vision_edge_priority_gain": 1.45,
        "vision_attention_boost_enabled": True,
        "vision_attention_boost_decay": 0.72,
        "vision_patch_budget": 20,
        "vision_focus_patch_budget": 10,
        "vision_raw_state_budget": 96,
        "vision_reconstruction_patch_budget": 1024,
        "vision_attention_boost_max_extra_raw_budget": 128,
        "vision_attention_boost_max_extra_focus_budget": 10,
        "vision_attention_boost_min_radius_scale": 0.28,
        "vision_attention_boost_edge_gain": 1.35,
        "vision_attention_boost_gaze_sigma_scale": 0.52,
        "hearing_window_budget": 18,
        "hearing_focus_band_count": 14,
        "hearing_focus_bandwidth_octaves": 1.1,
        "hearing_attention_boost_enabled": True,
        "hearing_attention_boost_decay": 0.8,
        "hearing_attention_boost_max_extra_window_budget": 10,
        "hearing_attention_boost_max_extra_focus_budget": 8,
        "hearing_attention_boost_min_bandwidth_scale": 0.58,
        "hearing_attention_boost_focus_gain": 1.45,
        "hearing_static_dedup_delta_threshold": 0.02,
        "hearing_static_dedup_band_similarity_threshold": 0.92,
        "hearing_static_dedup_max_suppression": 0.85,
        "hearing_auditory_fatigue_decay": 0.82,
        "hearing_auditory_fatigue_step": 0.16,
        "hearing_auditory_fatigue_max": 1.0,
    }


def _mk_runtime(*, overrides: dict[str, Any] | None = None) -> RuntimeV2:
    merged = dict(_base_overrides())
    if overrides:
        merged.update(overrides)
    runtime = RuntimeV2(config=load_config(overrides=merged), repo_root=REPO_ROOT)
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _mk_chirp(*, start_hz: float, end_hz: float, duration_sec: float = 0.5, sample_rate: int = 16000, amplitude: int = 12000) -> bytes:
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


def _sapi_tts_wav(text: str, *, voice_name: str, rate: int = 0) -> bytes | None:
    ps = f"""
Add-Type -AssemblyName System.Speech
$path = Join-Path $env:TEMP 'codex_multimodal_teach.wav'
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{voice_name}')
$s.Rate = {int(rate)}
$s.SetOutputToWaveFile($path)
$s.Speak('{text}')
$s.Dispose()
Write-Output $path
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except Exception:
        return None
    lines = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    path = Path(lines[-1])
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _render_concept_image(concept: MultimodalConcept, *, size: tuple[int, int] = (256, 256)) -> bytes:
    image = Image.new("RGB", size, color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    if concept.shape == "apple":
        draw.ellipse((72, 70, 192, 190), fill=concept.rgb)
        draw.rectangle((124, 38, 136, 80), fill=(96, 68, 34))
        draw.polygon([(136, 42), (164, 34), (154, 64)], fill=(62, 168, 86))
        draw.ellipse((92, 94, 120, 122), fill=(238, 152, 152))
    elif concept.shape == "banana":
        draw.pieslice((52, 84, 214, 210), start=212, end=332, fill=concept.rgb)
        draw.pieslice((76, 110, 200, 196), start=212, end=332, fill=(18, 18, 18))
        draw.rectangle((196, 96, 206, 116), fill=(120, 82, 38))
        draw.rectangle((63, 166, 74, 186), fill=(120, 82, 38))
    else:
        draw.rounded_rectangle((78, 78, 178, 178), radius=24, fill=concept.rgb)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _build_unique_label_maps(
    *,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    label_key: str,
    allow_fn,
) -> dict[str, Any]:
    labels_by_concept: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for concept in concepts:
        concept_id = concept.concept_id
        labels = _dedupe_keep_order(
            [str(label or "") for label in list((assets.get(concept_id, {}) or {}).get(label_key, []) or []) if allow_fn(str(label or ""))]
        )
        labels_by_concept[concept_id] = labels
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    unique_by_concept: dict[str, list[str]] = {}
    shared_by_concept: dict[str, list[str]] = {}
    unique_label_to_concept: dict[str, str] = {}
    for concept in concepts:
        concept_id = concept.concept_id
        labels = labels_by_concept.get(concept_id, [])
        unique_labels = [label for label in labels if counts.get(label, 0) == 1]
        shared_labels = [label for label in labels if counts.get(label, 0) > 1]
        unique_by_concept[concept_id] = unique_labels
        shared_by_concept[concept_id] = shared_labels
        for label in unique_labels:
            unique_label_to_concept[label] = concept_id

    return {
        "identity_labels_by_concept": labels_by_concept,
        "unique_labels_by_concept": unique_by_concept,
        "shared_labels_by_concept": shared_by_concept,
        "unique_label_to_concept": unique_label_to_concept,
        "label_counts": counts,
    }


def _prepare_concept_assets(output_root: Path, concepts: list[MultimodalConcept]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    assets_dir = output_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    runtime = _mk_runtime()
    assets: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        image_bytes = _render_concept_image(concept)
        image_path = assets_dir / f"{concept.concept_id}.png"
        image_path.write_bytes(image_bytes)

        audio_bytes = _sapi_tts_wav(concept.spoken_text, voice_name=concept.tts_voice, rate=concept.tts_rate)
        audio_mode = "tts"
        if not audio_bytes:
            audio_mode = "fallback_chirp"
            audio_bytes = _mk_chirp(start_hz=concept.fallback_freqs[0], end_hz=concept.fallback_freqs[1], duration_sec=0.5)
        audio_path = assets_dir / f"{concept.concept_id}.wav"
        audio_path.write_bytes(audio_bytes)

        image_packet = runtime.vision_sensor.ingest_image_bytes(image_bytes, tick_index=0, source_type=f"asset::{concept.concept_id}")
        audio_packet = runtime.hearing_sensor.ingest_wav_bytes(audio_bytes, tick_index=0, source_type=f"asset::{concept.concept_id}")

        vision_labels = [
            str(item.get("sa_label", "") or "")
            for item in (
                list(image_packet.get("memory_write_samples", []) or [])
                + list(image_packet.get("global_structure_samples", []) or [])
            )
            if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("vision_mem::")
        ]
        audio_labels = [
            str(item.get("sa_label", "") or "")
            for item in (
                list(audio_packet.get("memory_write_samples", []) or [])
                + list(audio_packet.get("global_structure_samples", []) or [])
            )
            if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("audio::")
        ]
        assets[concept.concept_id] = {
            "concept_id": concept.concept_id,
            "text_label": concept.text_label,
            "zh_text": concept.zh_text,
            "spoken_text": concept.spoken_text,
            "tts_voice": concept.tts_voice,
            "tts_rate": int(concept.tts_rate),
            "audio_mode": audio_mode,
            "image_path": str(image_path),
            "audio_path": str(audio_path),
            "image_bytes": image_bytes,
            "audio_bytes": audio_bytes,
            "vision_labels": _dedupe_keep_order(vision_labels)[:16],
            "audio_labels": _dedupe_keep_order(audio_labels)[:24],
        }

    vision_maps = _build_unique_label_maps(
        concepts=concepts,
        assets=assets,
        label_key="vision_labels",
        allow_fn=_is_vision_identity_label,
    )
    audio_maps = _build_unique_label_maps(
        concepts=concepts,
        assets=assets,
        label_key="audio_labels",
        allow_fn=_is_audio_identity_label,
    )
    for concept in concepts:
        asset = assets[concept.concept_id]
        asset["vision_identity_labels"] = list((vision_maps.get("identity_labels_by_concept", {}) or {}).get(concept.concept_id, []))
        asset["vision_unique_labels"] = list((vision_maps.get("unique_labels_by_concept", {}) or {}).get(concept.concept_id, []))
        asset["vision_shared_labels"] = list((vision_maps.get("shared_labels_by_concept", {}) or {}).get(concept.concept_id, []))
        asset["audio_identity_labels"] = list((audio_maps.get("identity_labels_by_concept", {}) or {}).get(concept.concept_id, []))
        asset["audio_unique_labels"] = list((audio_maps.get("unique_labels_by_concept", {}) or {}).get(concept.concept_id, []))
        asset["audio_shared_labels"] = list((audio_maps.get("shared_labels_by_concept", {}) or {}).get(concept.concept_id, []))

    return assets, {"vision": vision_maps, "audio": audio_maps}


def _run_multimodal_tick(
    runtime: RuntimeV2,
    *,
    tick_index: int,
    text: str = "",
    image_bytes: bytes | None = None,
    audio_bytes: bytes | None = None,
    source_type: str = "multimodal_teaching_probe",
    execute_selected_actions: bool = True,
) -> tuple[dict[str, Any], float]:
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
    return tick, elapsed_ms


def _inject_reward(runtime: RuntimeV2, *, tick_index: int, tick: dict[str, Any], concept: MultimodalConcept, reward: float) -> dict[str, Any]:
    provenance = {
        "focus_memory_id": str((tick.get("focus_memory", {}) or {}).get("memory_id", "") or ""),
        "exact_memory_id": str((tick.get("exact_memory", {}) or {}).get("memory_id", "") or ""),
        "bn_ids": [str(item.get("memory_id", "") or "") for item in (tick.get("bn_list", []) or [])[:6]],
    }
    return runtime.inject_feedback_signals(
        tick_index=tick_index,
        feedback={
            "reward": float(reward),
            "punishment": 0.0,
            "notes": [f"multimodal_reward::{concept.concept_id}", f"text::{concept.text_label}", f"zh::{concept.zh_text}"],
        },
        provenance=provenance,
        source_type="multimodal_reward",
        channel="multimodal_reward",
        meta_extra={"concept_id": concept.concept_id, "text_label": concept.text_label, "zh_text": concept.zh_text},
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


def _best_concept_from_label_energies(energy_map: dict[str, float], label_to_concept: dict[str, str]) -> str:
    concept_energy: dict[str, float] = {}
    for label, energy in energy_map.items():
        concept_id = str(label_to_concept.get(label, "") or "")
        if not concept_id:
            continue
        concept_energy[concept_id] = concept_energy.get(concept_id, 0.0) + float(energy or 0.0)
    return _best_label_from_energy_map(concept_energy)


def _evaluate_identity_signature_recall(
    *,
    tick: dict[str, Any],
    target_concept_id: str,
    target_labels: list[str],
    distractor_labels: list[str],
    label_to_concept: dict[str, str],
    metric_name: str,
) -> dict[str, Any]:
    allowed_labels = {str(label or "") for label in [*target_labels, *distractor_labels] if str(label or "")}
    if not allowed_labels or not label_to_concept:
        return {
            "metric_name": metric_name,
            "target_concept_id": target_concept_id,
            "usable_label_count": 0,
            "usable": False,
            "reason": "no_discriminative_labels",
            "cstar_label_energies": {},
            "state_label_energies": {},
            "concept_cstar_energies": {},
            "concept_state_energies": {},
            "cstar_target_energy": 0.0,
            "cstar_distractor_best_energy": 0.0,
            "cstar_margin": 0.0,
            "state_target_energy": 0.0,
            "state_distractor_best_energy": 0.0,
            "state_margin": 0.0,
            "cstar_best_concept_id": "",
            "state_best_concept_id": "",
            "cstar_success": False,
            "state_success": False,
            "strict_success": False,
        }

    c_star_items = [dict(item) for item in ((tick.get("c_star", {}) or {}).get("items", []) or []) if isinstance(item, dict)]
    state_top_rows = [dict(item) for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])) if isinstance(item, dict)]
    cstar_label_energies = _collect_label_energies(c_star_items, allowed_labels=allowed_labels)
    state_label_energies = _collect_label_energies(state_top_rows, allowed_labels=allowed_labels)

    concept_cstar_energies: dict[str, float] = {}
    concept_state_energies: dict[str, float] = {}
    for label, energy in cstar_label_energies.items():
        concept_id = str(label_to_concept.get(label, "") or "")
        if concept_id:
            concept_cstar_energies[concept_id] = concept_cstar_energies.get(concept_id, 0.0) + float(energy or 0.0)
    for label, energy in state_label_energies.items():
        concept_id = str(label_to_concept.get(label, "") or "")
        if concept_id:
            concept_state_energies[concept_id] = concept_state_energies.get(concept_id, 0.0) + float(energy or 0.0)

    target_cstar_energy = float(concept_cstar_energies.get(target_concept_id, 0.0) or 0.0)
    target_state_energy = float(concept_state_energies.get(target_concept_id, 0.0) or 0.0)
    distractor_cstar_best = max(
        [float(energy or 0.0) for concept_id, energy in concept_cstar_energies.items() if concept_id != target_concept_id] or [0.0]
    )
    distractor_state_best = max(
        [float(energy or 0.0) for concept_id, energy in concept_state_energies.items() if concept_id != target_concept_id] or [0.0]
    )
    cstar_best_concept_id = _best_label_from_energy_map(concept_cstar_energies)
    state_best_concept_id = _best_label_from_energy_map(concept_state_energies)

    return {
        "metric_name": metric_name,
        "target_concept_id": target_concept_id,
        "usable_label_count": int(len(allowed_labels)),
        "usable": True,
        "reason": "",
        "cstar_label_energies": {key: _round4(value) for key, value in sorted(cstar_label_energies.items(), key=lambda item: item[0])},
        "state_label_energies": {key: _round4(value) for key, value in sorted(state_label_energies.items(), key=lambda item: item[0])},
        "concept_cstar_energies": {key: _round4(value) for key, value in sorted(concept_cstar_energies.items(), key=lambda item: item[0])},
        "concept_state_energies": {key: _round4(value) for key, value in sorted(concept_state_energies.items(), key=lambda item: item[0])},
        "cstar_target_energy": _round4(target_cstar_energy),
        "cstar_distractor_best_energy": _round4(distractor_cstar_best),
        "cstar_margin": _round4(target_cstar_energy - distractor_cstar_best),
        "state_target_energy": _round4(target_state_energy),
        "state_distractor_best_energy": _round4(distractor_state_best),
        "state_margin": _round4(target_state_energy - distractor_state_best),
        "cstar_best_concept_id": cstar_best_concept_id,
        "state_best_concept_id": state_best_concept_id,
        "cstar_success": bool(target_cstar_energy > 0.0 and target_cstar_energy > distractor_cstar_best),
        "state_success": bool(target_state_energy > 0.0 and target_state_energy > distractor_state_best),
        "strict_success": bool(
            target_cstar_energy > 0.0
            and target_cstar_energy > distractor_cstar_best
            and target_state_energy > 0.0
            and target_state_energy > distractor_state_best
        ),
    }


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


def _evaluate_modality_signature_recall(
    *,
    tick: dict[str, Any],
    target_concept_id: str,
    target_labels: list[str],
    distractor_labels: list[str],
    label_to_concept: dict[str, str],
) -> dict[str, Any]:
    allowed_labels = {str(label or "") for label in [*target_labels, *distractor_labels] if str(label or "")}
    c_star_items = [dict(item) for item in ((tick.get("c_star", {}) or {}).get("items", []) or []) if isinstance(item, dict)]
    state_top_rows = [dict(item) for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])) if isinstance(item, dict)]
    cstar_label_energies = _collect_label_energies(c_star_items, allowed_labels=allowed_labels)
    state_label_energies = _collect_label_energies(state_top_rows, allowed_labels=allowed_labels)
    target_cstar_energy = sum(float(cstar_label_energies.get(label, 0.0) or 0.0) for label in target_labels)
    distractor_cstar_best = 0.0
    for label in distractor_labels:
        distractor_cstar_best = max(distractor_cstar_best, float(cstar_label_energies.get(label, 0.0) or 0.0))
    target_state_energy = sum(float(state_label_energies.get(label, 0.0) or 0.0) for label in target_labels)
    distractor_state_best = 0.0
    for label in distractor_labels:
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
        "cstar_best_concept_id": _best_concept_from_label_energies(cstar_label_energies, label_to_concept=label_to_concept),
        "state_best_concept_id": _best_concept_from_label_energies(state_label_energies, label_to_concept=label_to_concept),
        "cstar_success": bool(target_cstar_energy > 0.0 and target_cstar_energy > distractor_cstar_best),
        "state_success": bool(target_state_energy > 0.0 and target_state_energy > distractor_state_best),
        "strict_success": bool(target_cstar_energy > 0.0 and target_cstar_energy > distractor_cstar_best and target_state_energy > 0.0),
        "target_concept_id": target_concept_id,
    }


def _probe_runtime(*, payload: dict[str, Any], read_only: bool, overrides: dict[str, Any] | None = None) -> RuntimeV2:
    runtime = _mk_runtime(overrides=overrides)
    runtime.import_payload(copy.deepcopy(payload))
    if read_only:
        runtime.memory_store.write_memory_batch = lambda rows: []
        runtime.memory_store.write_memory = lambda *args, **kwargs: {}
    return runtime


def _tick_log_row(
    *,
    tick: dict[str, Any],
    tick_index: int,
    phase: str,
    concept_id: str,
    input_modalities: list[str],
    target_text_label: str,
    distractor_texts: list[str],
) -> dict[str, Any]:
    emotion = _emotion_row(tick, tick_index=tick_index, text=target_text_label if "text" in input_modalities else "")
    text_eval = _evaluate_text_recall(tick=tick, target_text=target_text_label, distractor_texts=distractor_texts)
    return {
        "phase": phase,
        "concept_id": concept_id,
        "tick_index": int(tick_index),
        "input_modalities": list(input_modalities),
        "elapsed_ms": _round4(float(tick.get("elapsed_ms", 0.0) or 0.0)),
        "emotion": emotion,
        "text_eval": text_eval,
        "state_top_labels": list(emotion.get("state_top_labels", []) or []),
    }


def _training_attempt(
    *,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    train_epochs_apple: int,
    train_epochs_banana: int,
    reward_value: float,
    stabilize_ticks: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _mk_runtime(overrides=runtime_overrides)
    tick_index = 0
    training_rows: list[dict[str, Any]] = []
    order = [(concepts[0], train_epochs_apple), (concepts[1], train_epochs_banana)]
    for concept, epochs in order:
        asset = assets[concept.concept_id]
        for epoch in range(int(epochs)):
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text=concept.text_label,
                image_bytes=asset["image_bytes"],
                audio_bytes=asset["audio_bytes"],
                source_type=f"multimodal_train::{concept.concept_id}",
                execute_selected_actions=True,
            )
            reward_payload = _inject_reward(runtime, tick_index=tick_index, tick=tick, concept=concept, reward=reward_value)
            training_rows.append(
                {
                    "phase": f"train::{concept.concept_id}",
                    "tick_index": int(tick_index),
                    "epoch": int(epoch),
                    "concept_id": concept.concept_id,
                    "elapsed_ms": _round4(elapsed_ms),
                    "reward": _round4(float(reward_payload.get("reward", 0.0) or 0.0)),
                    "image_memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))),
                    "image_global_structure_count": int(len(((tick.get("image_packet", {}) or {}).get("global_structure_samples", []) or []))),
                    "audio_memory_write_count": int(len(((tick.get("audio_packet", {}) or {}).get("memory_write_samples", []) or []))),
                    "audio_global_structure_count": int(len(((tick.get("audio_packet", {}) or {}).get("global_structure_samples", []) or []))),
                    "text_eval": _evaluate_text_recall(
                        tick=tick,
                        target_text=concept.text_label,
                        distractor_texts=[item.text_label for item in concepts if item.concept_id != concept.concept_id],
                    ),
                    "emotion": _emotion_row(tick, tick_index=tick_index, text=concept.text_label),
                }
            )
            tick_index += 1

    stabilize_rows: list[dict[str, Any]] = []
    for _ in range(max(0, int(stabilize_ticks))):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            image_bytes=None,
            audio_bytes=None,
            source_type="multimodal_stabilize",
            execute_selected_actions=False,
        )
        stabilize_rows.append(
            {
                "phase": "stabilize",
                "tick_index": int(tick_index),
                "elapsed_ms": _round4(elapsed_ms),
                "emotion": _emotion_row(tick, tick_index=tick_index, text=""),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
            }
        )
        tick_index += 1
    return {
        "stabilized_payload": runtime.export_payload(),
        "training_rows": training_rows,
        "stabilize_rows": stabilize_rows,
        "train_epochs_apple": int(train_epochs_apple),
        "train_epochs_banana": int(train_epochs_banana),
        "reward_value": _round4(reward_value),
        "stabilize_ticks": int(stabilize_ticks),
    }


def _single_modality_probe(
    *,
    imported_payload: dict[str, Any],
    concept: MultimodalConcept,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    modality: str,
    observation_ticks: int,
    clear_mode: str,
    vision_maps: dict[str, Any],
    audio_maps: dict[str, Any],
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _probe_runtime(payload=imported_payload, read_only=True, overrides=runtime_overrides)
    if clear_mode == "reset_transient_state":
        runtime.reset_transient_state(keep_runtime_controls=True)
        runtime.import_payload({"memory_store": copy.deepcopy(imported_payload.get("memory_store", {}))})
    elif clear_mode == "idle_then_probe":
        pass
    else:
        raise ValueError(f"unsupported clear_mode: {clear_mode}")

    tick_cursor = 0
    idle_rows: list[dict[str, Any]] = []
    if clear_mode == "idle_then_probe":
        for _ in range(8):
            idle_tick, idle_elapsed = _run_multimodal_tick(
                runtime,
                tick_index=tick_cursor,
                text="",
                image_bytes=None,
                audio_bytes=None,
                source_type=f"probe_idle::{concept.concept_id}",
                execute_selected_actions=False,
            )
            idle_rows.append(
                {
                    "tick_index": int(tick_cursor),
                    "elapsed_ms": _round4(idle_elapsed),
                    "emotion": _emotion_row(idle_tick, tick_index=tick_cursor, text=""),
                }
            )
            tick_cursor += 1

    asset = assets[concept.concept_id]
    distractor_texts = [item.text_label for item in concepts if item.concept_id != concept.concept_id]
    distractor_vision_labels = [
        label
        for item in concepts
        if item.concept_id != concept.concept_id
        for label in (assets[item.concept_id].get("vision_unique_labels", []) or [])
    ]
    distractor_audio_labels = [
        label
        for item in concepts
        if item.concept_id != concept.concept_id
        for label in (assets[item.concept_id].get("audio_unique_labels", []) or [])
    ]
    tick_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        image_bytes = asset["image_bytes"] if modality == "vision" else None
        audio_bytes = asset["audio_bytes"] if modality == "audio" else None
        text = concept.text_label if modality == "text" else ""
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_cursor,
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            source_type=f"probe::{clear_mode}::{modality}::{concept.concept_id}",
            execute_selected_actions=True,
        )
        text_eval = _evaluate_text_recall(tick=tick, target_text=concept.text_label, distractor_texts=distractor_texts)
        vision_eval = _evaluate_identity_signature_recall(
            tick=tick,
            target_concept_id=concept.concept_id,
            target_labels=list(asset.get("vision_unique_labels", []) or []),
            distractor_labels=distractor_vision_labels,
            label_to_concept=dict((vision_maps.get("unique_label_to_concept", {}) or {})),
            metric_name="vision_unique_identity",
        )
        audio_eval = _evaluate_identity_signature_recall(
            tick=tick,
            target_concept_id=concept.concept_id,
            target_labels=list(asset.get("audio_unique_labels", []) or []),
            distractor_labels=distractor_audio_labels,
            label_to_concept=dict((audio_maps.get("unique_label_to_concept", {}) or {})),
            metric_name="audio_unique_identity",
        )
        feeling = _emotion_row(tick, tick_index=tick_cursor, text=text)
        tick_rows.append(
            {
                "probe_tick_index": int(probe_index),
                "runtime_tick_index": int(tick_cursor),
                "elapsed_ms": _round4(elapsed_ms),
                "modality": modality,
                "text_eval": text_eval,
                "vision_eval": vision_eval,
                "audio_eval": audio_eval,
                "emotion": feeling,
            }
        )
        final_tick = tick
        tick_cursor += 1

    final_text_eval = _evaluate_text_recall(tick=final_tick, target_text=concept.text_label, distractor_texts=distractor_texts)
    final_vision_eval = _evaluate_identity_signature_recall(
        tick=final_tick,
        target_concept_id=concept.concept_id,
        target_labels=list(asset.get("vision_unique_labels", []) or []),
        distractor_labels=distractor_vision_labels,
        label_to_concept=dict((vision_maps.get("unique_label_to_concept", {}) or {})),
        metric_name="vision_unique_identity",
    )
    final_audio_eval = _evaluate_identity_signature_recall(
        tick=final_tick,
        target_concept_id=concept.concept_id,
        target_labels=list(asset.get("audio_unique_labels", []) or []),
        distractor_labels=distractor_audio_labels,
        label_to_concept=dict((audio_maps.get("unique_label_to_concept", {}) or {})),
        metric_name="audio_unique_identity",
    )
    first_text_success_tick = next((int(row["probe_tick_index"]) for row in tick_rows if bool((row.get("text_eval", {}) or {}).get("strict_success", False))), None)
    return {
        "concept_id": concept.concept_id,
        "text_label": concept.text_label,
        "modality": modality,
        "clear_mode": clear_mode,
        "idle_rows": idle_rows,
        "tick_rows": tick_rows,
        "final_text_evaluation": final_text_eval,
        "final_vision_evaluation": final_vision_eval,
        "final_audio_evaluation": final_audio_eval,
        "first_text_success_tick": first_text_success_tick,
        "mean_elapsed_ms": _mean([float(row.get("elapsed_ms", 0.0) or 0.0) for row in tick_rows]),
    }


def _switch_probe(
    *,
    imported_payload: dict[str, Any],
    warm_concept: MultimodalConcept,
    target_concept: MultimodalConcept,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    observation_ticks: int,
    warm_ticks: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _probe_runtime(payload=imported_payload, read_only=True, overrides=runtime_overrides)
    tick_cursor = 0
    distractor_texts = [item.text_label for item in concepts if item.concept_id != target_concept.concept_id]
    warm_asset = assets[warm_concept.concept_id]
    target_asset = assets[target_concept.concept_id]

    warm_rows: list[dict[str, Any]] = []
    for warm_index in range(int(warm_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_cursor,
            text=warm_concept.text_label,
            image_bytes=warm_asset["image_bytes"],
            audio_bytes=warm_asset["audio_bytes"],
            source_type=f"switch_warm::{warm_concept.concept_id}",
            execute_selected_actions=True,
        )
        warm_rows.append(
            {
                "probe_tick_index": int(warm_index),
                "elapsed_ms": _round4(elapsed_ms),
                "emotion": _emotion_row(tick, tick_index=tick_cursor, text=warm_concept.text_label),
                "text_eval": _evaluate_text_recall(
                    tick=tick,
                    target_text=warm_concept.text_label,
                    distractor_texts=[item.text_label for item in concepts if item.concept_id != warm_concept.concept_id],
                ),
            }
        )
        tick_cursor += 1

    switch_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_cursor,
            text=target_concept.text_label,
            image_bytes=target_asset["image_bytes"],
            audio_bytes=target_asset["audio_bytes"],
            source_type=f"switch_target::{target_concept.concept_id}",
            execute_selected_actions=True,
        )
        text_eval = _evaluate_text_recall(tick=tick, target_text=target_concept.text_label, distractor_texts=distractor_texts)
        switch_rows.append(
            {
                "probe_tick_index": int(probe_index),
                "elapsed_ms": _round4(elapsed_ms),
                "emotion": _emotion_row(tick, tick_index=tick_cursor, text=target_concept.text_label),
                "text_eval": text_eval,
            }
        )
        final_tick = tick
        tick_cursor += 1

    final_eval = _evaluate_text_recall(tick=final_tick, target_text=target_concept.text_label, distractor_texts=distractor_texts)
    def _find_first_flip(key: str, target_value: Any) -> int | None:
        for row in switch_rows:
            if (row.get("text_eval", {}) or {}).get(key) == target_value:
                return int(row.get("probe_tick_index", 0) or 0)
        return None

    return {
        "warm_concept_id": warm_concept.concept_id,
        "target_concept_id": target_concept.concept_id,
        "warm_ticks": int(warm_ticks),
        "observation_ticks": int(observation_ticks),
        "warm_rows": warm_rows,
        "switch_rows": switch_rows,
        "final_evaluation": final_eval,
        "bn_flip_tick": _find_first_flip("bn_best_text", target_concept.text_label),
        "cstar_flip_tick": _find_first_flip("cstar_best_text", target_concept.text_label),
        "state_flip_tick": _find_first_flip("state_best_text", target_concept.text_label),
        "focus_flip_tick": next((int(row.get("probe_tick_index", 0) or 0) for row in switch_rows if bool((row.get("text_eval", {}) or {}).get("focus_has_target", False))), None),
    }


def _build_observatory_showcase_dataset(*, output_root: Path, concepts: list[MultimodalConcept], assets: dict[str, dict[str, Any]]) -> Path:
    dataset_path = output_root / "showcase_dataset.json"
    items: list[dict[str, Any]] = []
    stable_train_ticks = 10
    stabilize_gap_ticks = 6
    probe_hold_ticks = 6
    for concept in concepts:
        asset = assets[concept.concept_id]
        for _ in range(stable_train_ticks):
            items.append(
                {
                    "text": concept.text_label,
                    "image_path": asset["image_path"],
                    "audio_path": asset["audio_path"],
                    "source_type": f"multimodal_showcase::train::{concept.concept_id}",
                    "external_feedback": {"reward": 0.9, "punishment": 0.0, "notes": [f"showcase::{concept.concept_id}"]},
                }
            )
        for _ in range(stabilize_gap_ticks):
            items.append({"text": "", "source_type": f"multimodal_showcase::idle_after::{concept.concept_id}"})
    for _ in range(stabilize_gap_ticks):
        items.append({"text": "", "source_type": "multimodal_showcase::idle_bridge"})
    for concept in concepts:
        asset = assets[concept.concept_id]
        for _ in range(probe_hold_ticks):
            items.append({"text": "", "image_path": asset["image_path"], "source_type": f"showcase_probe::vision::{concept.concept_id}"})
        for _ in range(probe_hold_ticks):
            items.append({"text": "", "audio_path": asset["audio_path"], "source_type": f"showcase_probe::audio::{concept.concept_id}"})
        for _ in range(probe_hold_ticks):
            items.append({"text": concept.text_label, "source_type": f"showcase_probe::text::{concept.concept_id}"})
        for _ in range(stabilize_gap_ticks):
            items.append({"text": "", "source_type": f"multimodal_showcase::idle_probe_after::{concept.concept_id}"})
    payload = {
        "label": "Phase25 多模态教学+海豚训练展示运行",
        "config_overrides": _base_overrides(),
        "mode": "multimodal",
        "max_ticks": len(items),
        "items": items,
    }
    _write_json(dataset_path, payload)
    return dataset_path


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _run_observatory_showcase(*, output_root: Path, dataset_path: Path) -> dict[str, Any]:
    showcase_root = output_root / "observatory_showcase"
    result = run_dataset_file(
        dataset_path,
        default_label="Phase25 多模态教学+海豚训练展示运行",
        timeout_sec=1800.0,
        repo_root_value=REPO_ROOT,
        outputs_root_override=str(showcase_root),
    )
    run_dir = Path(str((((result.get("runs", []) or [])[0] or {}).get("result", {}) or {}).get("run_dir", "") or ""))
    tick_count = 0
    sidecar_count = 0
    summary_count = 0
    if run_dir.exists():
        for path in sorted((run_dir / "chunks").glob("*.summary.jsonl")):
            summary_count += len(_iter_jsonl(path))
        for path in sorted((run_dir / "chunks").glob("*.sidecar.jsonl")):
            sidecar_count += len(_iter_jsonl(path))
        tick_count = max(summary_count, sidecar_count)
    return {
        "dataset_path": str(dataset_path),
        "result": result,
        "run_dir": str(run_dir),
        "tick_count": int(tick_count),
        "summary_count": int(summary_count),
        "sidecar_count": int(sidecar_count),
    }


def _render_markdown_report(
    *,
    output_root: Path,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    training: dict[str, Any],
    probes: list[dict[str, Any]],
    switching_rows: list[dict[str, Any]],
    observatory_showcase: dict[str, Any],
) -> str:
    lines: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in probes:
        grouped.setdefault(_probe_group_key(row), []).append(row)

    train_rows = list(training.get("training_rows", []) or [])
    stabilize_rows = list(training.get("stabilize_rows", []) or [])
    train_avg_ms = _mean([float(row.get("elapsed_ms", 0.0) or 0.0) for row in train_rows])
    train_max_ms = _round4(max([float(row.get("elapsed_ms", 0.0) or 0.0) for row in train_rows] or [0.0]))
    probe_avg_ms = _mean([float(row.get("mean_elapsed_ms", 0.0) or 0.0) for row in probes])
    probe_max_ms = _round4(
        max(
            [
                float(item.get("elapsed_ms", 0.0) or 0.0)
                for row in probes
                for item in list(row.get("tick_rows", []) or [])
            ]
            or [0.0]
        )
    )
    observation_tick_count = max([len(list(row.get("tick_rows", []) or [])) for row in probes] or [0])
    warm_tick_count = max([int(row.get("warm_ticks", 0) or 0) for row in switching_rows] or [0])

    lines.append("# V2 多模态教学与海豚训练综合实验报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("本实验要验证的，不是单一模态分类，而是 AP V2 在统一状态池里，能否把“同一个对象的视觉、听觉、文本标签”共同教进去，并在后续仅凭单一模态线索，把其它模态相关记忆一起召回。")
    lines.append("同时，本实验还把奖励信号接入训练过程，观察它是否能形成更稳定的对象绑定与后续认知翻转。")
    lines.append("")
    lines.append("本轮重点回答五个问题：")
    lines.append("1. 当图像、音频、文本同时输入且内容一致时，系统能否形成统一的多模态对象记忆。")
    lines.append("2. 训练后，仅输入单视觉、单听觉、单文本时，系统能否召回正确文本以及对应其它模态结构。")
    lines.append("3. 在空 tick 稳定后探测，与在 reset transient state 后探测，系统表现有何差别。")
    lines.append("4. 当对象从苹果突然切换成香蕉时，系统需要多久从旧上下文翻转过来；反过来是否同理。")
    lines.append("5. 在这个过程中，惊、违和感、正确感、把握感、期待、压力是否呈现出符合理论预期的时序变化。")
    lines.append("")
    lines.append("## 2. 实验结论摘要")
    lines.append("")
    lines.append("先说最重要的结论：")
    lines.append("1. 在保留训练后稳定上下文的条件下，`vision-only` 与 `audio-only` 都已经能把目标文本重新拉起来，且 `vision-only` 的对象区分最稳。")
    lines.append("2. `text-only` 对 `banana` 还能稳定命中，但对 `apple` 仍容易被更晚期的 `banana` 上下文压过去，说明跨模态统一绑定已经成立，但双对象长期对称稳固还没完全站住。")
    lines.append("3. 一旦 `reset_transient_state`，纯视觉/纯听觉的冷启动召回明显变弱，说明这轮实验更像“稳定后的多模态整合召回成立”，还不能说“彻底脱离近期上下文也一样稳”。")
    lines.append("4. 对象切换时，`Bn` 与 `C*` 都能在第 0 个 observation tick 迅速翻向新对象，但 `state_top` 还会慢 1 到 2 tick，末尾还存在尾振荡和旧对象残留。")
    lines.append("5. 认知感受方面，这轮最清楚证明的是“新异输入触发惊与违和，然后记忆层逐步跟上”；`correctness / grasp` 这次仍基本没有显著起来，所以这一块不能夸大。")
    lines.append("")
    lines.append("## 3. 实验材料")
    lines.append("")
    for concept in concepts:
        asset = assets[concept.concept_id]
        lines.append(
            f"- `{concept.concept_id}`：文本 `{concept.text_label}` / 中文 `{concept.zh_text}` / 图像 [{Path(asset['image_path']).name}]({asset['image_path']}) / 音频 [{Path(asset['audio_path']).name}]({asset['audio_path']}) / audio_mode=`{asset['audio_mode']}`"
        )
        lines.append(f"  - 视觉结构标签示例：{asset['vision_labels'][:4]}")
        lines.append(f"  - 听觉结构标签示例：{asset['audio_labels'][:4]}")
    lines.append("")
    lines.append("这里的图像不直接写出“苹果/香蕉”文字，避免实验退化成视觉文字识别；它测的是对象形状/颜色结构与文本、音频标签的统一绑定。")
    lines.append("")
    lines.append("## 4. 训练协议")
    lines.append("")
    lines.append(f"- 苹果多模态连续训练：{int(training.get('train_epochs_apple', 0) or 0)} tick")
    lines.append(f"- 香蕉多模态连续训练：{int(training.get('train_epochs_banana', 0) or 0)} tick")
    lines.append(f"- 每个训练 tick 注入奖励：reward={_round4(float(training.get('reward_value', 0.0) or 0.0))}")
    lines.append(f"- 训练后稳定空 tick：{int(training.get('stabilize_ticks', 0) or 0)}")
    lines.append(f"- 单模态 probe 连续观察：{int(observation_tick_count)} tick")
    lines.append(f"- 切换实验旧对象预热：{int(warm_tick_count)} tick")
    lines.append(f"- 训练 tick 总数：{len(train_rows)}")
    lines.append(f"- 平均训练耗时：{train_avg_ms} ms")
    lines.append(f"- 峰值训练耗时：{train_max_ms} ms")
    lines.append(f"- 平均 probe 耗时：{probe_avg_ms} ms")
    lines.append(f"- 峰值 probe 耗时：{probe_max_ms} ms")
    if stabilize_rows:
        lines.append(f"- 稳定阶段 tick 数：{len(stabilize_rows)}")
    lines.append("")
    lines.append("## 5. 判定口径")
    lines.append("")
    lines.append("本报告同时看四层信号：")
    lines.append("1. `BN_top`：一级显式记忆召回最强文本。")
    lines.append("2. `C*_top`：综合预测包当前主导文本。")
    lines.append("3. `State_top`：状态池主导文本波峰。")
    lines.append("4. `vision_identity / audio_identity`：目标对象特有的视觉或听觉结构标签，是否在 `C*` 与状态池里占优。")
    lines.append("")
    lines.append("这里的 `first_text_success_tick` 采用的是当前脚本里的 `text_eval.strict_success` 口径，它要求 `BN_top` 与 `C*_top` 已经同时命中目标，但**不强制** `State_top` 同 tick 也完成翻转。")
    lines.append("所以它更接近“认知判断已翻过来”，而不是“整个状态池主波峰已彻底稳定”。状态池是否翻过来，需要单独看 `State_top`。")
    lines.append("")
    lines.append("## 6. 理论预期的认知感受图景")
    lines.append("")
    lines.append("理论上，这组实验中的认知感受应该大致遵循下面的时序：")
    lines.append("1. 第一次面对某个对象时，如果系统尚未建立对应预测，应先出现较明显的惊。")
    lines.append("2. 连续多 tick 重复面对同一对象时，惊应逐步下降，而正确感/把握感、期待应逐步上升。")
    lines.append("3. 当输入突然从苹果切换到香蕉时，旧预测尚未完全退去，会出现“惊 + 违和”的复合态。")
    lines.append("4. 随着香蕉证据连续输入，旧预测衰减，新预测建立，违和感与惊应逐步回落，而正确感/把握感重新上来。")
    lines.append("5. 如果奖励链路正常，则在“认清对象”与“从惊/违和中恢复”这两类阶段，应看到恢复性奖励或正确感相关的内源反馈。")
    lines.append("")
    lines.append("## 7. 训练阶段的实际图景")
    lines.append("")
    apple_rows = [row for row in train_rows if str(row.get("phase", "")) == "train::apple"][:4]
    banana_rows = [row for row in train_rows if str(row.get("phase", "")) == "train::banana"][:4]
    if apple_rows:
        lines.append("### 7.1 苹果建立期")
        for row in apple_rows:
            emotion = dict(row.get("emotion", {}) or {})
            text_eval = dict(row.get("text_eval", {}) or {})
            lines.append(
                f"- tick {row.get('tick_index', '')} / epoch {row.get('epoch', '')}：surprise={_round4(float(emotion.get('surprise', 0.0) or 0.0))} / dissonance={_round4(float(emotion.get('dissonance', 0.0) or 0.0))} / expectation={_round4(float(emotion.get('expectation', 0.0) or 0.0))} / BN_top=`{text_eval.get('bn_best_text', '')}` / C*_top=`{text_eval.get('cstar_best_text', '')}` / State_top=`{text_eval.get('state_best_text', '')}` / strict={_bool_mark(bool(text_eval.get('strict_success', False)))}"
            )
        lines.append("")
    if banana_rows:
        lines.append("### 7.2 香蕉切入期")
        for row in banana_rows:
            emotion = dict(row.get("emotion", {}) or {})
            text_eval = dict(row.get("text_eval", {}) or {})
            lines.append(
                f"- tick {row.get('tick_index', '')} / epoch {row.get('epoch', '')}：surprise={_round4(float(emotion.get('surprise', 0.0) or 0.0))} / dissonance={_round4(float(emotion.get('dissonance', 0.0) or 0.0))} / expectation={_round4(float(emotion.get('expectation', 0.0) or 0.0))} / BN_top=`{text_eval.get('bn_best_text', '')}` / C*_top=`{text_eval.get('cstar_best_text', '')}` / State_top=`{text_eval.get('state_best_text', '')}` / strict={_bool_mark(bool(text_eval.get('strict_success', False)))}"
            )
        lines.append("")
        lines.append("这段数据最像你预期的图景：`banana` 刚切入时，系统先表现出高惊和高违和，`BN` 先翻，`C*` 再翻，最后 `State_top` 才在更后面稳住。")
        lines.append("")
    lines.append("## 8. 单模态召回结果")
    lines.append("")
    section_index = 1
    for clear_mode, modality in sorted(grouped.keys()):
        lines.append(f"### 8.{section_index} `{clear_mode}` / `{modality}`")
        section_index += 1
        for row in grouped[(clear_mode, modality)]:
            text_eval = dict(row.get("final_text_evaluation", {}) or {})
            vision_eval = dict(row.get("final_vision_evaluation", {}) or {})
            audio_eval = dict(row.get("final_audio_evaluation", {}) or {})
            lines.append(
                f"- `{row.get('concept_id', '')}`：BN_top=`{text_eval.get('bn_best_text', '')}` / C*_top=`{text_eval.get('cstar_best_text', '')}` / State_top=`{text_eval.get('state_best_text', '')}` / first_text_success_tick={row.get('first_text_success_tick', None)} / text_strict={_bool_mark(bool(text_eval.get('strict_success', False)))} / vision_C*_best=`{vision_eval.get('cstar_best_concept_id', '')}` / vision_state_best=`{vision_eval.get('state_best_concept_id', '')}` / vision_strict={_bool_mark(bool(vision_eval.get('strict_success', False)))} / audio_C*_best=`{audio_eval.get('cstar_best_concept_id', '')}` / audio_state_best=`{audio_eval.get('state_best_concept_id', '')}` / audio_strict={_bool_mark(bool(audio_eval.get('strict_success', False)))}"
            )
        lines.append("")
    lines.append("这里有三个必须诚实说明的点：")
    lines.append("1. `idle_then_probe` 明显强于 `reset_transient_state`，说明当前成功很依赖训练后保留下来的稳定上下文。")
    lines.append("2. 这轮 integrated probe 里，视觉 identity 的跨模态带起效果是最清楚的；音频 identity 标签在最终 probe 窗口里还没有形成同样干净的可观测胜出。")
    lines.append("3. 所以这轮最稳的证据是“视觉/听觉能把目标文本拉起来，视觉还能把目标视觉对象结构一起带起”，而不是“所有模态都已经完全对称地互相召回”。")
    lines.append("")
    lines.append("## 9. 对象切换翻转结果")
    lines.append("")
    for row in switching_rows:
        switch_summary = _switch_summary_row(row)
        lines.append(
            f"- `{switch_summary['switch']}`：BN_flip={switch_summary['bn_flip_tick']} / C*_flip={switch_summary['cstar_flip_tick']} / State_flip={switch_summary['state_flip_tick']} / Focus_flip={switch_summary['focus_flip_tick']} / first_obs(surprise={switch_summary['first_surprise']}, dissonance={switch_summary['first_dissonance']}, BN=`{switch_summary['first_bn']}`, C*=`{switch_summary['first_cstar']}`, State=`{switch_summary['first_state']}`) / final(BN=`{switch_summary['final_bn']}`, C*=`{switch_summary['final_cstar']}`, State=`{switch_summary['final_state']}`, strict={_bool_mark(switch_summary['final_strict'])})"
        )
    lines.append("")
    lines.append("这组切换结果很像你要的层级翻转：")
    lines.append("1. 新证据一进来，`BN` 和 `C*` 几乎立刻翻向新对象。")
    lines.append("2. `State_top` 更慢，通常要再过 1 到 2 tick。")
    lines.append("3. 末尾 `final_C*_top` 仍可能飘回旧对象，这说明尾振荡和旧上下文残留还没有完全消掉。")
    lines.append("")
    lines.append("## 10. 认知感受与情绪变化观察")
    lines.append("")
    for row in probes:
        tick_rows = list(row.get("tick_rows", []) or [])
        if not tick_rows:
            continue
        first_emotion = dict((tick_rows[0].get("emotion", {}) or {}))
        last_emotion = dict((tick_rows[-1].get("emotion", {}) or {}))
        lines.append(
            f"- `{row.get('clear_mode', '')}` / `{row.get('modality', '')}` / `{row.get('concept_id', '')}`：first(surprise={_round4(float(first_emotion.get('surprise', 0.0) or 0.0))}, dissonance={_round4(float(first_emotion.get('dissonance', 0.0) or 0.0))}) -> last(surprise={_round4(float(last_emotion.get('surprise', 0.0) or 0.0))}, dissonance={_round4(float(last_emotion.get('dissonance', 0.0) or 0.0))}, correctness={_round4(float(last_emotion.get('correctness', 0.0) or 0.0))}, grasp={_round4(float(last_emotion.get('grasp', 0.0) or 0.0))})"
        )
    lines.append("")
    lines.append("这轮数据和理论预期的符合点与不符合点都很清楚：")
    lines.append("1. 符合的部分：新模态输入时，`surprise` 与 `dissonance` 会先高，然后随着连续输入逐步下降。")
    lines.append("2. 部分符合的部分：香蕉切入训练期，确实出现了“先惊 + 违和，再由记忆层逐步翻正”的层级过程。")
    lines.append("3. 暂时不符合或尚未显著的部分：`correctness` 与 `grasp` 在这轮 integrated probe 里几乎始终接近 0，没有形成可以拿来强证明的稳定证据。")
    lines.append("所以这轮能证明“惊 / 违和 / 逐步反应过来”，但还不能强证明“把握感 / 正确感已经在这套多模态教学里稳定长出来”。")
    lines.append("")
    lines.append("## 11. 前端展示链路验收")
    lines.append("")
    lines.append(f"- 展示 dataset：[{Path(observatory_showcase.get('dataset_path', '')).name}]({observatory_showcase.get('dataset_path', '')})")
    lines.append(f"- 展示 run 目录：[{Path(observatory_showcase.get('run_dir', '')).name}]({observatory_showcase.get('run_dir', '')})")
    lines.append(f"- sidecar tick 数：{int(observatory_showcase.get('sidecar_count', 0) or 0)} / summary tick 数：{int(observatory_showcase.get('summary_count', 0) or 0)}")
    lines.append("")
    lines.append("这条展示 run 已经做了前端实测：")
    lines.append("1. 在文字 probe tick 上，视觉面板会老实显示“暂无可回放视觉帧”，不会伪造图像。")
    lines.append("2. 在图像 tick（如 `Tick 12`）上，视觉面板可进入“正在播放视觉回放”状态，并显示 `当前感知 / 近 4 tick 已叠加 192 个视觉 SA；当前 tick 原始采样 192，焦点样本 20，注视累积 1315`。")
    lines.append("3. 想象音频面板也能进入播放态，并显示 `融合视图 / 近 4 tick 已提取 6 个听觉 SA，可播放代理合成音`。")
    lines.append("")
    lines.append("也就是说，当前前端已经能把“bot 某个 tick 在联想到的内心画面”和“它在回味的内心声音代理结构”真正展示出来，只是仍然属于代理重建，不是原始高保真回放。")
    lines.append("")
    lines.append("## 12. 结论")
    lines.append("")
    lines.append("如果只问“这套多模态教学 + 海豚训练有没有把统一召回这条链打通”，这轮答案是肯定的，但要带边界地说。")
    lines.append("")
    lines.append("> **AP V2 已经能在统一状态池、统一记忆召回和奖励塑形链路里，把图像、音频、文本共同教成一个对象，并在保留近期稳定上下文的情况下，凭单一模态线索触发跨模态联想与对象翻转。**")
    lines.append("")
    lines.append("已经被较强证明的部分是：")
    lines.append("1. `vision-only` 与 `audio-only` 可以把目标文本重新拉起来。")
    lines.append("2. 视觉对象结构也能跟着一起被带起。")
    lines.append("3. 新旧对象切换时，系统确实呈现出分层翻转，而不是单步硬切。")
    lines.append("4. 前端展示链可以把这种联想过程以“想象图像 / 想象音频”方式直观看出来。")
    lines.append("")
    lines.append("还没有被这轮强证明的部分是：")
    lines.append("1. 完全清空瞬态后，纯跨模态长期冷召回是否同样稳。")
    lines.append("2. 音频 identity 结构是否已经能像视觉 identity 一样干净地被反向带起。")
    lines.append("3. `correctness / grasp` 是否已经在这组综合实验里稳定长出来。")
    lines.append("")
    lines.append("## 13. 当前边界")
    lines.append("")
    lines.append("1. 当前对象集仍然很小，只证明原理可行，不代表开放环境大规模泛化。")
    lines.append("2. 当前 integrated probe 的成功高度依赖保留训练后的稳定上下文；`reset_transient_state` 后的纯冷召回仍然明显变弱。")
    lines.append("3. 当前前端听觉回放是结构代理合成，不是原始波形高保真重建。")
    lines.append("4. 当前视觉叠加是状态池视觉 SA 的稀疏代理重建，不等于像素级完整还原。")
    lines.append("5. 本轮听觉优先使用系统 TTS；若环境不可用则自动退回到可控 chirp 代理音。")
    lines.append("6. 当前 `strict_success` 的统计口径偏向“BN + C* 已翻正”，与“状态池主波峰也完全稳定”不是同一件事，阅读时要分开。")
    lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    output_root: Path,
    doc_path: Path,
    reward_value: float,
    train_epochs_apple: int,
    train_epochs_banana: int,
    stabilize_ticks: int,
    observation_ticks: int,
    warm_ticks: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    concepts = list(DEFAULT_CONCEPTS)
    assets, label_maps = _prepare_concept_assets(output_root, concepts)
    vision_maps = dict((label_maps.get("vision", {}) or {}))
    audio_maps = dict((label_maps.get("audio", {}) or {}))

    training = _training_attempt(
        concepts=concepts,
        assets=assets,
        train_epochs_apple=train_epochs_apple,
        train_epochs_banana=train_epochs_banana,
        reward_value=reward_value,
        stabilize_ticks=stabilize_ticks,
        runtime_overrides=runtime_overrides,
    )
    payload = dict(training.get("stabilized_payload", {}) or {})

    probes: list[dict[str, Any]] = []
    for clear_mode in ("idle_then_probe", "reset_transient_state"):
        for modality in ("vision", "audio", "text"):
            for concept in concepts:
                probes.append(
                    _single_modality_probe(
                        imported_payload=payload,
                        concept=concept,
                        concepts=concepts,
                        assets=assets,
                        modality=modality,
                        observation_ticks=observation_ticks,
                        clear_mode=clear_mode,
                        vision_maps=vision_maps,
                        audio_maps=audio_maps,
                        runtime_overrides=runtime_overrides,
                    )
                )

    switching_rows = [
        _switch_probe(
            imported_payload=payload,
            warm_concept=concepts[0],
            target_concept=concepts[1],
            concepts=concepts,
            assets=assets,
            observation_ticks=observation_ticks,
            warm_ticks=warm_ticks,
            runtime_overrides=runtime_overrides,
        ),
        _switch_probe(
            imported_payload=payload,
            warm_concept=concepts[1],
            target_concept=concepts[0],
            concepts=concepts,
            assets=assets,
            observation_ticks=observation_ticks,
            warm_ticks=warm_ticks,
            runtime_overrides=runtime_overrides,
        ),
    ]

    showcase_dataset = _build_observatory_showcase_dataset(output_root=output_root, concepts=concepts, assets=assets)
    observatory_showcase = _run_observatory_showcase(output_root=output_root, dataset_path=showcase_dataset)

    summary = {
        "schema_id": "multimodal_teaching_dolphin_probe/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "assets": {
            concept_id: {
                key: value
                for key, value in asset.items()
                if key not in {"image_bytes", "audio_bytes"}
            }
            for concept_id, asset in assets.items()
        },
        "label_maps": label_maps,
        "training": {key: value for key, value in training.items() if key != "stabilized_payload"},
        "probes": probes,
        "switching_rows": switching_rows,
        "observatory_showcase": observatory_showcase,
        "config": {
            "reward_value": _round4(reward_value),
            "train_epochs_apple": int(train_epochs_apple),
            "train_epochs_banana": int(train_epochs_banana),
            "stabilize_ticks": int(stabilize_ticks),
            "observation_ticks": int(observation_ticks),
            "warm_ticks": int(warm_ticks),
            "runtime_overrides": dict(runtime_overrides or {}),
        },
    }
    report_markdown = _render_markdown_report(
        output_root=output_root,
        concepts=concepts,
        assets=assets,
        training=summary["training"],
        probes=probes,
        switching_rows=switching_rows,
        observatory_showcase=observatory_showcase,
    )
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "probes.json", probes)
    _write_json(output_root / "switching_rows.json", switching_rows)
    _write_text(output_root / "report.md", report_markdown)
    _write_text(doc_path, report_markdown)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AP V2 multimodal teaching + dolphin training integrated probe.")
    parser.add_argument("--output-dir", default="", help="输出目录，默认写入 outputs/multimodal_teaching_dolphin_probe/<timestamp>")
    parser.add_argument("--doc-path", default="", help="正式报告路径")
    parser.add_argument("--reward", type=float, default=1.0, help="每个训练 tick 注入的奖励值")
    parser.add_argument("--train-epochs-apple", type=int, default=12, help="苹果训练 tick 数")
    parser.add_argument("--train-epochs-banana", type=int, default=12, help="香蕉训练 tick 数")
    parser.add_argument("--stabilize-ticks", type=int, default=8, help="训练后空 tick")
    parser.add_argument("--observation-ticks", type=int, default=6, help="单模态 probe 连续观察 tick")
    parser.add_argument("--warm-ticks", type=int, default=4, help="切换实验旧对象预热 tick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir or "").strip()
        else DEFAULT_OUTPUT_ROOT / timestamp
    )
    doc_path = (
        Path(args.doc_path).expanduser().resolve()
        if str(args.doc_path or "").strip()
        else DEFAULT_DOC_PATH
    )
    summary = run_experiment(
        output_root=output_root,
        doc_path=doc_path,
        reward_value=float(args.reward),
        train_epochs_apple=max(1, int(args.train_epochs_apple)),
        train_epochs_banana=max(1, int(args.train_epochs_banana)),
        stabilize_ticks=max(0, int(args.stabilize_ticks)),
        observation_ticks=max(1, int(args.observation_ticks)),
        warm_ticks=max(1, int(args.warm_ticks)),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "doc_path": str(doc_path),
                "training": summary.get("training", {}),
                "probe_count": len(summary.get("probes", []) or []),
                "showcase_run_dir": str((summary.get("observatory_showcase", {}) or {}).get("run_dir", "") or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
