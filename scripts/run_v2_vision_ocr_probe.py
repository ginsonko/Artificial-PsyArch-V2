# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vision_ocr_probe"
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\LHANDW.TTF"),
    Path(r"C:\Windows\Fonts\segoesc.ttf"),
    Path(r"C:\Windows\Fonts\BRUSHSCI.TTF"),
    Path(r"C:\Windows\Fonts\comic.ttf"),
]


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


@dataclass(frozen=True)
class OCRPair:
    pair_id: str
    glyph: str
    text_label: str
    rotate_deg: float


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_handwritten_image(pair: OCRPair, *, size: tuple[int, int] = (256, 128)) -> bytes:
    canvas = Image.new("RGB", size, color=(12, 12, 12))
    glyph_layer = Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(glyph_layer)
    font = _load_font(104)
    bbox = draw.textbbox((0, 0), pair.glyph, font=font, stroke_width=3)
    text_w = max(1, int(bbox[2] - bbox[0]))
    text_h = max(1, int(bbox[3] - bbox[1]))
    x = int((size[0] - text_w) / 2 - bbox[0])
    y = int((size[1] - text_h) / 2 - bbox[1])
    draw.text(
        (x, y),
        pair.glyph,
        font=font,
        fill=(245, 245, 245, 255),
        stroke_width=3,
        stroke_fill=(220, 220, 220, 255),
    )
    rotated = glyph_layer.rotate(pair.rotate_deg, resample=Image.Resampling.BICUBIC, expand=False)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), rotated).convert("RGB")
    buf = BytesIO()
    canvas.save(buf, format="PNG")
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
        "vision_edge_candidate_gain": 1.9,
        "vision_edge_priority_gain": 1.45,
        "vision_attention_boost_enabled": True,
        "vision_attention_boost_decay": 0.72,
        "vision_patch_budget": 16,
        "vision_focus_patch_budget": 8,
        "vision_raw_state_budget": 64,
        "vision_attention_boost_max_extra_raw_budget": 192,
        "vision_attention_boost_max_extra_focus_budget": 8,
        "vision_attention_boost_min_radius_scale": 0.28,
        "vision_attention_boost_edge_gain": 1.35,
        "vision_attention_boost_gaze_sigma_scale": 0.52,
    }


def _training_overrides(*, raw_budget: int, patch_budget: int, focus_budget: int) -> dict[str, Any]:
    return {
        **_base_overrides(),
        "vision_raw_state_budget": int(min(256, raw_budget)),
        "vision_patch_budget": int(patch_budget),
        "vision_focus_patch_budget": int(focus_budget),
        "vision_reconstruction_patch_budget": int(max(1024, min(2048, raw_budget * 4))),
    }


def _probe_overrides(*, raw_budget: int, patch_budget: int, focus_budget: int) -> dict[str, Any]:
    return {
        **_base_overrides(),
        "vision_raw_state_budget": int(min(256, raw_budget)),
        "vision_patch_budget": int(patch_budget),
        "vision_focus_patch_budget": int(focus_budget),
        "vision_reconstruction_patch_budget": int(max(1024, min(2048, raw_budget * 4))),
    }


def _budget_triplet(raw_budget: int) -> tuple[int, int, int]:
    bounded_raw = max(32, min(256, int(raw_budget)))
    patch_budget = max(8, min(24, int(round(bounded_raw / 16.0))))
    focus_budget = max(4, min(12, int(round(patch_budget / 2.0))))
    return int(bounded_raw), int(patch_budget), int(focus_budget)


def _build_runtime(*, overrides: dict[str, Any]) -> RuntimeV2:
    runtime = RuntimeV2(config=load_config(overrides=overrides), repo_root=REPO_ROOT)
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _run_multimodal_tick(
    runtime: RuntimeV2,
    *,
    tick_index: int,
    text: str,
    image_bytes: bytes | None,
    source_type: str,
    execute_selected_actions: bool = True,
) -> tuple[dict[str, Any], float]:
    text_packet = runtime.text_sensor.ingest(text, tick_index=tick_index, source_type=source_type)
    image_packet = (
        runtime.vision_sensor.ingest_image_bytes(image_bytes, tick_index=tick_index, source_type=source_type)
        if image_bytes is not None
        else None
    )
    started = time.perf_counter()
    tick = runtime.process_multimodal_tick(
        tick_index=tick_index,
        text_packet=text_packet,
        image_packet=image_packet,
        source_type=source_type,
    )
    runtime_action_effects = {"gaze_center_before": runtime.vision_gaze_snapshot(), "gaze_center_after": runtime.vision_gaze_snapshot(), "moved": False, "applied_actions": []}
    if execute_selected_actions:
        selected_actions = list(((tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or []))
        runtime_action_effects = runtime.apply_selected_actions(selected_actions, runtime_tick=tick)
    tick["runtime_action_effects"] = runtime_action_effects
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.set_last_logic_ms(elapsed_ms)
    return tick, elapsed_ms


def _inject_reward(runtime: RuntimeV2, *, tick_index: int, tick: dict[str, Any], pair: OCRPair, reward: float) -> dict[str, Any]:
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
            "notes": [f"ocr_reward::{pair.pair_id}", f"label::{pair.text_label}"],
        },
        provenance=provenance,
        source_type="vision_ocr_reward",
        channel="vision_ocr_reward",
        meta_extra={
            "pair_id": pair.pair_id,
            "target_text_label": pair.text_label,
            "target_glyph": pair.glyph,
        },
    )


def _filter_exact_bn(bn_list: list[dict[str, Any]], allowed_texts: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bn_list or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("memory_kind", "") or "") != "exact_external":
            continue
        text = str(row.get("text", "") or "").strip()
        if text in allowed_texts:
            rows.append(
                {
                    "memory_id": str(row.get("memory_id", "") or ""),
                    "text": text,
                    "score": _round4(float(row.get("score", 0.0) or 0.0)),
                    "tick_index": int(row.get("tick_index", -1) or -1),
                }
            )
    rows.sort(key=lambda item: (-float(item["score"]), -int(item["tick_index"]), item["memory_id"]))
    return rows


def _cstar_text_energies(c_star: dict[str, Any], allowed_texts: set[str]) -> dict[str, float]:
    energies: dict[str, float] = {}
    for item in (c_star.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("sa_label", "") or "")
        if not label.startswith("text::"):
            continue
        text = label.split("::", 1)[1]
        if text not in allowed_texts:
            continue
        energies[text] = _round4(energies.get(text, 0.0) + float(item.get("energy", 0.0) or 0.0))
    return energies


def _best_label_from_energy_map(energies: dict[str, float]) -> str:
    if not energies:
        return ""
    ordered = sorted(energies.items(), key=lambda item: (-float(item[1]), item[0]))
    return str(ordered[0][0])


def _evaluate_probe(
    *,
    tick: dict[str, Any],
    target_text: str,
    distractor_texts: list[str],
) -> dict[str, Any]:
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
    focus_units = [str(item or "") for item in ((tick.get("a_focus", {}) or {}).get("focus_units", []) or []) if str(item or "")]
    state_top_rows = [dict(item) for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])) if isinstance(item, dict)]
    state_top_labels = [str(item.get("sa_label", "") or "") for item in state_top_rows]
    state_text_energies: dict[str, float] = {}
    for row in state_top_rows:
        label = str(row.get("sa_label", "") or "")
        if not label.startswith("text::"):
            continue
        text = label.split("::", 1)[1]
        if text not in allowed_texts:
            continue
        state_text_energies[text] = _round4(state_text_energies.get(text, 0.0) + float(row.get("energy", 0.0) or 0.0))
    state_best_text = _best_label_from_energy_map(state_text_energies)
    state_distractor_best = 0.0
    for text in distractor_texts:
        state_distractor_best = max(state_distractor_best, float(state_text_energies.get(text, 0.0) or 0.0))
    state_target_energy = float(state_text_energies.get(target_text, 0.0) or 0.0)
    focus_best_text = target_text if target_text in focus_units else next((text for text in distractor_texts if text in focus_units), "")
    return {
        "bn_exact_ranked": exact_bn,
        "bn_best_text": bn_best_text,
        "bn_target_rank": int(bn_target_rank),
        "bn_target_score": _round4(bn_target_score),
        "bn_success": bool(bn_best_text == target_text and bn_target_rank == 1),
        "cstar_text_energies": {key: _round4(value) for key, value in sorted(cstar_energies.items(), key=lambda item: item[0])},
        "cstar_best_text": cstar_best_text,
        "cstar_target_energy": _round4(target_energy),
        "cstar_distractor_best_energy": _round4(distractor_best),
        "cstar_margin": _round4(target_energy - distractor_best),
        "cstar_success": bool(cstar_best_text == target_text and target_energy > distractor_best),
        "focus_units": focus_units,
        "focus_has_target": bool(target_text in focus_units),
        "focus_best_text": focus_best_text,
        "focus_success": bool(target_text in focus_units),
        "state_text_energies": {key: _round4(value) for key, value in sorted(state_text_energies.items(), key=lambda item: item[0])},
        "state_best_text": state_best_text,
        "state_target_energy": _round4(state_target_energy),
        "state_distractor_best_energy": _round4(state_distractor_best),
        "state_margin": _round4(state_target_energy - state_distractor_best),
        "state_top_contains_target_text": bool(f"text::{target_text}" in state_top_labels),
        "state_success": bool(state_best_text == target_text and state_target_energy > state_distractor_best),
        "strict_success": bool(
            bn_best_text == target_text
            and bn_target_rank == 1
            and cstar_best_text == target_text
            and target_energy > distractor_best
        ),
    }


def _probe_session(
    *,
    imported_payload: dict[str, Any],
    import_mode: str,
    pair: OCRPair,
    all_pairs: list[OCRPair],
    image_bytes: bytes,
    raw_budget: int,
    observation_ticks: int,
) -> dict[str, Any]:
    raw_budget, patch_budget, focus_budget = _budget_triplet(raw_budget)
    runtime = _build_runtime(overrides=_probe_overrides(raw_budget=raw_budget, patch_budget=patch_budget, focus_budget=focus_budget))
    if import_mode == "full":
        runtime.import_payload(copy.deepcopy(imported_payload))
    elif import_mode == "memory_only":
        runtime.import_payload({"memory_store": copy.deepcopy(imported_payload.get("memory_store", {}))})
    else:
        raise ValueError(f"unsupported import mode: {import_mode}")

    tick_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_index,
            text="",
            image_bytes=image_bytes,
            source_type=f"vision_ocr_probe::{pair.pair_id}",
        )
        eval_row = _evaluate_probe(
            tick=tick,
            target_text=pair.text_label,
            distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
        )
        tick_rows.append(
            {
                "probe_tick_index": probe_index,
                "elapsed_ms": _round4(elapsed_ms),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
                "raw_sample_count": int((tick.get("image_packet", {}) or {}).get("total_patch_count", 0) or 0),
                "memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))),
                "focus_memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("focus_priority_samples", []) or []))),
                "bn_best_text": eval_row["bn_best_text"],
                "bn_target_rank": int(eval_row["bn_target_rank"]),
                "cstar_best_text": eval_row["cstar_best_text"],
                "cstar_target_energy": _round4(eval_row["cstar_target_energy"]),
                "cstar_margin": _round4(eval_row["cstar_margin"]),
                "state_best_text": str(eval_row["state_best_text"]),
                "state_margin": _round4(eval_row["state_margin"]),
                "focus_has_target": bool(eval_row["focus_has_target"]),
                "strict_success": bool(eval_row["strict_success"]),
            }
        )
        final_tick = tick

    final_eval = _evaluate_probe(
        tick=final_tick,
        target_text=pair.text_label,
        distractor_texts=[item.text_label for item in all_pairs if item.pair_id != pair.pair_id],
    )
    return {
        "pair_id": pair.pair_id,
        "target_text_label": pair.text_label,
        "import_mode": import_mode,
        "raw_budget": raw_budget,
        "patch_budget": patch_budget,
        "focus_budget": focus_budget,
        "observation_ticks": int(observation_ticks),
        "tick_rows": tick_rows,
        "final_evaluation": final_eval,
        "mean_elapsed_ms": _round4(sum(float(row["elapsed_ms"]) for row in tick_rows) / max(1, len(tick_rows))),
    }


def _run_training_attempt(
    *,
    pairs: list[OCRPair],
    image_map: dict[str, bytes],
    train_epochs: int,
    train_raw_budget: int,
    train_patch_budget: int,
    train_focus_budget: int,
    stabilize_ticks: int,
    reward_value: float,
) -> dict[str, Any]:
    runtime = _build_runtime(
        overrides=_training_overrides(
            raw_budget=train_raw_budget,
            patch_budget=train_patch_budget,
            focus_budget=train_focus_budget,
        )
    )
    tick_index = 0
    training_rows: list[dict[str, Any]] = []
    for epoch in range(int(train_epochs)):
        for pair in pairs:
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text=pair.text_label,
                image_bytes=image_map[pair.pair_id],
                source_type=f"vision_ocr_train::{pair.pair_id}",
            )
            reward_payload = _inject_reward(runtime, tick_index=tick_index, tick=tick, pair=pair, reward=reward_value)
            training_eval = _evaluate_probe(
                tick=tick,
                target_text=pair.text_label,
                distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
            )
            training_rows.append(
                {
                    "tick_index": tick_index,
                    "epoch": epoch,
                    "pair_id": pair.pair_id,
                    "glyph": pair.glyph,
                    "text_label": pair.text_label,
                    "elapsed_ms": _round4(elapsed_ms),
                    "raw_sample_count": int((tick.get("image_packet", {}) or {}).get("total_patch_count", 0) or 0),
                    "memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))),
                    "focus_memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("focus_priority_samples", []) or []))),
                    "bn_best_text": training_eval["bn_best_text"],
                    "cstar_best_text": training_eval["cstar_best_text"],
                    "reward_injected": _round4(float(reward_payload.get("reward", 0.0) or 0.0)),
                    "memory_count_after_tick": int(tick.get("memory_count", 0) or 0),
                }
            )
            tick_index += 1

    stabilize_rows: list[dict[str, Any]] = []
    for _ in range(int(stabilize_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            image_bytes=None,
            source_type="vision_ocr_stabilize",
        )
        stabilize_rows.append(
            {
                "tick_index": tick_index,
                "elapsed_ms": _round4(elapsed_ms),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
                "top_preview": [str(item.get("display_text", "") or "") for item in ((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])[:6]],
            }
        )
        tick_index += 1

    stabilized_payload = runtime.export_payload()
    acceptance_rows: list[dict[str, Any]] = []
    all_success = True
    for pair in pairs:
        probe_row = _probe_session(
            imported_payload=stabilized_payload,
            import_mode="memory_only",
            pair=pair,
            all_pairs=pairs,
            image_bytes=image_map[pair.pair_id],
            raw_budget=256,
            observation_ticks=5,
        )
        acceptance_rows.append(probe_row)
        if not bool((probe_row.get("final_evaluation", {}) or {}).get("strict_success", False)):
            all_success = False

    return {
        "train_epochs": int(train_epochs),
        "train_raw_budget": int(train_raw_budget),
        "train_patch_budget": int(train_patch_budget),
        "train_focus_budget": int(train_focus_budget),
        "stabilize_ticks": int(stabilize_ticks),
        "reward_value": _round4(reward_value),
        "training_rows": training_rows,
        "stabilize_rows": stabilize_rows,
        "acceptance_rows": acceptance_rows,
        "accepted": bool(all_success),
        "stabilized_payload": stabilized_payload,
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
            }
        )
    return summary


def _run_sampling_sweep(
    *,
    payload: dict[str, Any],
    pairs: list[OCRPair],
    image_map: dict[str, bytes],
    raw_budgets: list[int],
    observation_ticks_list: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw_budget in raw_budgets:
        for observation_ticks in observation_ticks_list:
            pair_runs: list[dict[str, Any]] = []
            for pair in pairs:
                session = _probe_session(
                    imported_payload=payload,
                    import_mode="memory_only",
                    pair=pair,
                    all_pairs=pairs,
                    image_bytes=image_map[pair.pair_id],
                    raw_budget=raw_budget,
                    observation_ticks=observation_ticks,
                )
                pair_runs.append(session)

            strict_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("strict_success", False)))
            bn_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("bn_success", False)))
            cstar_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("cstar_success", False)))
            focus_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("focus_has_target", False)))
            margins = [float((row.get("final_evaluation", {}) or {}).get("cstar_margin", 0.0) or 0.0) for row in pair_runs]
            mean_elapsed_ms = sum(float(row.get("mean_elapsed_ms", 0.0) or 0.0) for row in pair_runs) / max(1, len(pair_runs))
            rows.append(
                {
                    "raw_budget": int(raw_budget),
                    "observation_ticks": int(observation_ticks),
                    "strict_accuracy": _round4(strict_hits / max(1, len(pair_runs))),
                    "bn_accuracy": _round4(bn_hits / max(1, len(pair_runs))),
                    "cstar_accuracy": _round4(cstar_hits / max(1, len(pair_runs))),
                    "focus_accuracy": _round4(focus_hits / max(1, len(pair_runs))),
                    "mean_cstar_margin": _round4(sum(margins) / max(1, len(margins))),
                    "mean_elapsed_ms_per_tick": _round4(mean_elapsed_ms),
                    "pair_runs": pair_runs,
                }
            )

    frontier: list[dict[str, Any]] = []
    for raw_budget in raw_budgets:
        candidates = [row for row in rows if int(row.get("raw_budget", 0) or 0) == int(raw_budget)]
        candidates.sort(key=lambda row: int(row.get("observation_ticks", 0) or 0))
        chosen = next((row for row in candidates if float(row.get("strict_accuracy", 0.0) or 0.0) >= 1.0), None)
        frontier.append(
            {
                "raw_budget": int(raw_budget),
                "min_observation_ticks_for_strict_success": int(chosen.get("observation_ticks", 0) or 0) if chosen else None,
                "best_strict_accuracy": _round4(max((float(row.get("strict_accuracy", 0.0) or 0.0) for row in candidates), default=0.0)),
                "best_cstar_accuracy": _round4(max((float(row.get("cstar_accuracy", 0.0) or 0.0) for row in candidates), default=0.0)),
            }
        )
    return {"rows": rows, "frontier": frontier}


def _render_report_markdown(
    *,
    selected_attempt: dict[str, Any],
    main_probe_rows: list[dict[str, Any]],
    sweep: dict[str, Any],
) -> str:
    acceptance_summary = _summarize_acceptance(list(selected_attempt.get("acceptance_rows", []) or []))
    lines: list[str] = []
    lines.append("# V2 视觉 OCR-like 预实验报告")
    lines.append("")
    lines.append("## 1. 训练设定")
    lines.append(
        f"- 训练轮次：{int(selected_attempt.get('train_epochs', 0) or 0)} epoch（每个 epoch 交替呈现两张手写风格数字图像）"
    )
    lines.append(
        f"- 训练采样：raw={int(selected_attempt.get('train_raw_budget', 0) or 0)} / memory={int(selected_attempt.get('train_patch_budget', 0) or 0)} / focus={int(selected_attempt.get('train_focus_budget', 0) or 0)}"
    )
    lines.append(f"- 奖励信号：每个正确图像-文本共现 tick 注入 reward={_round4(float(selected_attempt.get('reward_value', 0.0) or 0.0))}")
    lines.append(f"- 稳定空 tick：{int(selected_attempt.get('stabilize_ticks', 0) or 0)}")
    lines.append("")
    lines.append("## 2. 主实验结论")
    for row in main_probe_rows:
        final_eval = dict(row.get("final_evaluation", {}) or {})
        lines.append(
            f"- `{row.get('pair_id', '')}` -> 目标文本 `{row.get('target_text_label', '')}`："
            f"BN_top=`{final_eval.get('bn_best_text', '')}`，"
            f"C*_top=`{final_eval.get('cstar_best_text', '')}`，"
            f"margin={_round4(float(final_eval.get('cstar_margin', 0.0) or 0.0))}，"
            f"strict_success={bool(final_eval.get('strict_success', False))}"
        )
    lines.append("")
    lines.append("## 3. 接受门槛检查（训练后 1024 raw / 4 tick 冷探测）")
    for row in acceptance_summary:
        lines.append(
            f"- `{row['pair_id']}`: BN_rank={row['bn_target_rank']} / C*_top=`{row['cstar_best_text']}` / "
            f"focus_has_target={row['focus_has_target']} / strict_success={row['strict_success']}"
        )
    lines.append("")
    lines.append("## 4. 采样率-准确率扫描")
    for row in sweep.get("frontier", []) or []:
        lines.append(
            f"- raw={int(row.get('raw_budget', 0) or 0)} 时，首次达到 strict_accuracy=1.0 所需 observation_ticks="
            f"{row.get('min_observation_ticks_for_strict_success', None)}；best_strict_accuracy={row.get('best_strict_accuracy', 0.0)}"
        )
    lines.append("")
    lines.append("## 5. 解释边界")
    lines.append("- 这个实验若成功，证明的是 AP 视觉稀疏采样特征可以和文本标签建立可召回联结，属于 OCR-like 的初步关联识别。")
    lines.append("- 它还不能直接等价于通用 OCR，更不能证明跨字体、跨噪声、跨布局的完整泛化能力。")
    lines.append("- 奖励信号已纳入训练流程，但本轮默认没有做无奖励对照，因此证明的是“带奖励的配对训练可行”，不是“奖励是唯一原因”。")
    lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    output_root: Path,
    train_plan_candidates: list[dict[str, int]],
    stabilize_ticks: int,
    reward_value: float,
    raw_budgets: list[int],
    observation_ticks_list: list[int],
) -> dict[str, Any]:
    pairs = [
        OCRPair(pair_id="digit_3", glyph="3", text_label="three", rotate_deg=-6.0),
        OCRPair(pair_id="digit_8", glyph="8", text_label="eight", rotate_deg=5.0),
    ]
    image_dir = output_root / "generated_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, bytes] = {}
    image_manifest: list[dict[str, Any]] = []
    for pair in pairs:
        raw = _render_handwritten_image(pair)
        image_map[pair.pair_id] = raw
        image_path = image_dir / f"{pair.pair_id}.png"
        image_path.write_bytes(raw)
        image_manifest.append(
            {
                "pair_id": pair.pair_id,
                "glyph": pair.glyph,
                "text_label": pair.text_label,
                "image_path": str(image_path),
            }
        )

    attempts: list[dict[str, Any]] = []
    selected_attempt: dict[str, Any] | None = None
    for plan in train_plan_candidates:
        attempt = _run_training_attempt(
            pairs=pairs,
            image_map=image_map,
            train_epochs=int(plan["train_epochs"]),
            train_raw_budget=int(plan["train_raw_budget"]),
            train_patch_budget=int(plan["train_patch_budget"]),
            train_focus_budget=int(plan["train_focus_budget"]),
            stabilize_ticks=stabilize_ticks,
            reward_value=reward_value,
        )
        attempts.append(
            {
                key: value
                for key, value in attempt.items()
                if key != "stabilized_payload"
            }
        )
        if bool(attempt.get("accepted", False)):
            selected_attempt = attempt
            break
        if selected_attempt is None:
            selected_attempt = attempt

    if selected_attempt is None:
        raise RuntimeError("未能生成任何训练尝试结果")

    stabilized_payload = dict(selected_attempt.get("stabilized_payload", {}) or {})
    main_probe_rows = []
    for pair in pairs:
        main_probe_rows.append(
            _probe_session(
                imported_payload=stabilized_payload,
                import_mode="memory_only",
                pair=pair,
                all_pairs=pairs,
                image_bytes=image_map[pair.pair_id],
                raw_budget=256,
                observation_ticks=6,
            )
        )

    sweep = _run_sampling_sweep(
        payload=stabilized_payload,
        pairs=pairs,
        image_map=image_map,
        raw_budgets=raw_budgets,
        observation_ticks_list=observation_ticks_list,
    )

    report_markdown = _render_report_markdown(
        selected_attempt=selected_attempt,
        main_probe_rows=main_probe_rows,
        sweep=sweep,
    )
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_manifest": image_manifest,
        "attempts": attempts,
        "selected_attempt": {
            key: value
            for key, value in selected_attempt.items()
            if key != "stabilized_payload"
        },
        "main_probe_rows": main_probe_rows,
        "sampling_sweep": sweep,
    }
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "report.md", report_markdown)
    _write_json(output_root / "selected_training_rows.json", list(selected_attempt.get("training_rows", []) or []))
    _write_json(output_root / "sampling_sweep_rows.json", list(sweep.get("rows", []) or []))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 视觉 OCR-like 关联实验与采样率扫描")
    parser.add_argument("--output-dir", default="", help="输出目录，默认写入 outputs/vision_ocr_probe/<timestamp>")
    parser.add_argument("--stabilize-ticks", type=int, default=10, help="训练后用于稳定状态池的空 tick 数")
    parser.add_argument("--reward", type=float, default=1.0, help="每个训练 tick 注入的奖励值")
    parser.add_argument("--raw-budgets", default="64,128,256,512,1024", help="采样扫描的 raw_state_budget 列表")
    parser.add_argument("--observation-ticks", default="1,2,4,8,12", help="每个采样率下连续观察 tick 数列表")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir).expanduser() if str(args.output_dir or "").strip() else DEFAULT_OUTPUT_ROOT / timestamp
    raw_budgets = [max(16, min(256, int(item.strip()))) for item in str(args.raw_budgets or "").split(",") if item.strip()]
    observation_ticks_list = [max(1, int(item.strip())) for item in str(args.observation_ticks or "").split(",") if item.strip()]
    train_plan_candidates = [
        {"train_epochs": 12, "train_raw_budget": 256, "train_patch_budget": 16, "train_focus_budget": 8},
        {"train_epochs": 16, "train_raw_budget": 256, "train_patch_budget": 16, "train_focus_budget": 8},
    ]
    summary = run_experiment(
        output_root=output_root,
        train_plan_candidates=train_plan_candidates,
        stabilize_ticks=max(0, int(args.stabilize_ticks)),
        reward_value=float(args.reward),
        raw_budgets=raw_budgets,
        observation_ticks_list=observation_ticks_list,
    )
    print(json.dumps({"output_dir": str(output_root), "selected_attempt": summary.get("selected_attempt", {}), "frontier": summary.get("sampling_sweep", {}).get("frontier", [])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
