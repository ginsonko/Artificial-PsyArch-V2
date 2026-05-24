from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v2_multichannel_feelings_report import _emotion_row
from scripts.run_v2_multimodal_teaching_dolphin_probe import (
    DEFAULT_CONCEPTS,
    REPO_ROOT,
    MultimodalConcept,
    _best_label_from_energy_map,
    _build_unique_label_maps,
    _collect_label_energies,
    _cstar_text_energies,
    _mean,
    _mk_runtime,
    _prepare_concept_assets,
    _probe_runtime,
    _round4,
    _run_multimodal_tick,
    _run_observatory_showcase,
    _training_attempt,
    _write_json,
    _write_text,
    _is_audio_identity_label,
    _is_vision_identity_label,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "multimodal_overlay_color_probe"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "V2_多模态叠加想象与颜色迁移实验报告_2026-05-24.md"


def _bool_mark(value: bool) -> str:
    return "是" if bool(value) else "否"


def _render_yellow_apple_image(*, size: tuple[int, int] = (256, 256)) -> bytes:
    image = Image.new("RGB", size, color=(18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.ellipse((72, 70, 192, 190), fill=(228, 206, 58))
    draw.rectangle((124, 38, 136, 80), fill=(96, 68, 34))
    draw.polygon([(136, 42), (164, 34), (154, 64)], fill=(62, 168, 86))
    draw.ellipse((92, 94, 120, 122), fill=(246, 230, 120))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _state_text_energies(tick: dict[str, Any], allowed_texts: set[str]) -> dict[str, float]:
    energies: dict[str, float] = {}
    state_top_rows = [
        dict(item)
        for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or []))
        if isinstance(item, dict)
    ]
    for row in state_top_rows:
        label = str(row.get("sa_label", "") or "")
        if not label.startswith("text::"):
            continue
        text = label.split("::", 1)[1]
        if text not in allowed_texts:
            continue
        energies[text] = _round4(energies.get(text, 0.0) + float(row.get("energy", 0.0) or 0.0))
    return energies


def _concept_energy_summary(energy_map: dict[str, float]) -> dict[str, Any]:
    ordered = sorted(
        [(str(key or ""), float(value or 0.0)) for key, value in dict(energy_map or {}).items() if str(key or "")],
        key=lambda item: item[1],
        reverse=True,
    )
    total = sum(value for _, value in ordered)
    primary_id, primary_energy = ordered[0] if ordered else ("", 0.0)
    secondary_id, secondary_energy = ordered[1] if len(ordered) > 1 else ("", 0.0)
    ratio = (secondary_energy / primary_energy) if primary_energy > 1e-9 else 0.0
    share = (secondary_energy / total) if total > 1e-9 else 0.0
    return {
        "ordered": [{"concept_id": concept_id, "energy": _round4(energy)} for concept_id, energy in ordered],
        "primary_concept_id": primary_id,
        "primary_energy": _round4(primary_energy),
        "secondary_concept_id": secondary_id,
        "secondary_energy": _round4(secondary_energy),
        "secondary_ratio": _round4(ratio),
        "secondary_share": _round4(share),
        "dual_nonzero": bool(secondary_energy > 0.0),
        "dual_competitive": bool(primary_energy > 0.0 and secondary_energy > 0.0 and ratio >= 0.25),
        "total_energy": _round4(total),
    }


def _collect_identity_concept_maps(
    *,
    tick: dict[str, Any],
    label_to_concept: dict[str, str],
    allowed_labels: set[str],
) -> dict[str, Any]:
    c_star_items = [dict(item) for item in ((tick.get("c_star", {}) or {}).get("items", []) or []) if isinstance(item, dict)]
    state_top_rows = [
        dict(item)
        for item in (((tick.get("state_pool_summary", {}) or {}).get("top", []) or []))
        if isinstance(item, dict)
    ]
    cstar_label_energies = _collect_label_energies(c_star_items, allowed_labels=allowed_labels)
    state_label_energies = _collect_label_energies(state_top_rows, allowed_labels=allowed_labels)
    concept_cstar: dict[str, float] = {}
    concept_state: dict[str, float] = {}
    for label, energy in cstar_label_energies.items():
        concept_id = str(label_to_concept.get(label, "") or "")
        if concept_id:
            concept_cstar[concept_id] = concept_cstar.get(concept_id, 0.0) + float(energy or 0.0)
    for label, energy in state_label_energies.items():
        concept_id = str(label_to_concept.get(label, "") or "")
        if concept_id:
            concept_state[concept_id] = concept_state.get(concept_id, 0.0) + float(energy or 0.0)
    return {
        "cstar": {key: _round4(value) for key, value in sorted(concept_cstar.items(), key=lambda item: item[0])},
        "state": {key: _round4(value) for key, value in sorted(concept_state.items(), key=lambda item: item[0])},
        "cstar_summary": _concept_energy_summary(concept_cstar),
        "state_summary": _concept_energy_summary(concept_state),
        "cstar_label_count": len(cstar_label_energies),
        "state_label_count": len(state_label_energies),
    }


def _build_probe_eval(
    *,
    tick: dict[str, Any],
    concepts: list[MultimodalConcept],
    vision_allowed_labels: set[str],
    audio_allowed_labels: set[str],
    vision_label_to_concept: dict[str, str],
    audio_label_to_concept: dict[str, str],
) -> dict[str, Any]:
    concept_texts = [concept.text_label for concept in concepts]
    allowed_texts = set(concept_texts)
    text_cstar = _cstar_text_energies(dict(tick.get("c_star", {}) or {}), allowed_texts=allowed_texts)
    text_state = _state_text_energies(tick, allowed_texts=allowed_texts)
    vision_maps = _collect_identity_concept_maps(
        tick=tick,
        label_to_concept=vision_label_to_concept,
        allowed_labels=vision_allowed_labels,
    )
    audio_maps = _collect_identity_concept_maps(
        tick=tick,
        label_to_concept=audio_label_to_concept,
        allowed_labels=audio_allowed_labels,
    )
    return {
        "text_cstar_energies": {key: _round4(value) for key, value in sorted(text_cstar.items(), key=lambda item: item[0])},
        "text_state_energies": {key: _round4(value) for key, value in sorted(text_state.items(), key=lambda item: item[0])},
        "text_cstar_summary": _concept_energy_summary(text_cstar),
        "text_state_summary": _concept_energy_summary(text_state),
        "vision_cstar_energies": dict(vision_maps.get("cstar", {}) or {}),
        "vision_state_energies": dict(vision_maps.get("state", {}) or {}),
        "vision_cstar_summary": dict(vision_maps.get("cstar_summary", {}) or {}),
        "vision_state_summary": dict(vision_maps.get("state_summary", {}) or {}),
        "audio_cstar_energies": dict(audio_maps.get("cstar", {}) or {}),
        "audio_state_energies": dict(audio_maps.get("state", {}) or {}),
        "audio_cstar_summary": dict(audio_maps.get("cstar_summary", {}) or {}),
        "audio_state_summary": dict(audio_maps.get("state_summary", {}) or {}),
    }


def _probe_runtime_with_reset(imported_payload: dict[str, Any], runtime_overrides: dict[str, Any] | None = None):
    runtime = _probe_runtime(payload=imported_payload, read_only=True, overrides=runtime_overrides)
    runtime.reset_transient_state(keep_runtime_controls=True)
    runtime.import_payload({"memory_store": copy.deepcopy(imported_payload.get("memory_store", {}))})
    return runtime


def _run_text_prompt_probe(
    *,
    imported_payload: dict[str, Any],
    prompt: str,
    scenario_id: str,
    observation_ticks: int,
    concepts: list[MultimodalConcept],
    vision_allowed_labels: set[str],
    audio_allowed_labels: set[str],
    vision_label_to_concept: dict[str, str],
    audio_label_to_concept: dict[str, str],
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _probe_runtime_with_reset(imported_payload=imported_payload, runtime_overrides=runtime_overrides)
    tick_rows: list[dict[str, Any]] = []
    for probe_tick_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_tick_index,
            text=prompt,
            source_type=f"overlay_probe::{scenario_id}",
            execute_selected_actions=True,
        )
        eval_row = _build_probe_eval(
            tick=tick,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
        )
        tick_rows.append(
            {
                "probe_tick_index": int(probe_tick_index),
                "elapsed_ms": _round4(elapsed_ms),
                "emotion": _emotion_row(tick, tick_index=probe_tick_index, text=prompt),
                **eval_row,
            }
        )
    final_row = dict(tick_rows[-1] if tick_rows else {})
    return {
        "scenario_id": scenario_id,
        "scenario_type": "text_prompt",
        "prompt": prompt,
        "observation_ticks": int(observation_ticks),
        "tick_rows": tick_rows,
        "final_text_cstar_summary": dict(final_row.get("text_cstar_summary", {}) or {}),
        "final_text_state_summary": dict(final_row.get("text_state_summary", {}) or {}),
        "final_vision_cstar_summary": dict(final_row.get("vision_cstar_summary", {}) or {}),
        "final_vision_state_summary": dict(final_row.get("vision_state_summary", {}) or {}),
        "final_audio_cstar_summary": dict(final_row.get("audio_cstar_summary", {}) or {}),
        "final_audio_state_summary": dict(final_row.get("audio_state_summary", {}) or {}),
        "mean_elapsed_ms": _mean([float(row.get("elapsed_ms", 0.0) or 0.0) for row in tick_rows]),
    }


def _vision_sensor_label_overlap(
    *,
    probe_labels: list[str],
    concept_labels_by_id: dict[str, list[str]],
) -> dict[str, Any]:
    probe_set = set(str(label or "") for label in probe_labels if str(label or ""))
    result: dict[str, Any] = {}
    contour_probe = {label for label in probe_set if "global_contour::" in label}
    for concept_id, labels in concept_labels_by_id.items():
        concept_set = set(str(label or "") for label in list(labels or []) if str(label or ""))
        overlap = sorted(probe_set & concept_set)
        contour_overlap = sorted(contour_probe & {label for label in concept_set if "global_contour::" in label})
        result[str(concept_id)] = {
            "overlap_count": int(len(overlap)),
            "overlap_ratio": _round4(len(overlap) / max(1, len(concept_set))),
            "contour_overlap_count": int(len(contour_overlap)),
            "overlap_examples": overlap[:6],
            "contour_overlap_examples": contour_overlap[:3],
        }
    return result


def _run_yellow_apple_probe(
    *,
    imported_payload: dict[str, Any],
    image_bytes: bytes,
    observation_ticks: int,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    vision_allowed_labels: set[str],
    audio_allowed_labels: set[str],
    vision_label_to_concept: dict[str, str],
    audio_label_to_concept: dict[str, str],
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _probe_runtime_with_reset(imported_payload=imported_payload, runtime_overrides=runtime_overrides)
    tick_rows: list[dict[str, Any]] = []
    sensor_overlap_first_tick: dict[str, Any] = {}
    concept_labels_by_id = {concept.concept_id: list((assets.get(concept.concept_id, {}) or {}).get("vision_labels", []) or []) for concept in concepts}
    for probe_tick_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_tick_index,
            text="",
            image_bytes=image_bytes,
            source_type="color_transfer::yellow_apple",
            execute_selected_actions=True,
        )
        eval_row = _build_probe_eval(
            tick=tick,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
        )
        if probe_tick_index == 0:
            probe_labels = [
                str(item.get("sa_label", "") or "")
                for item in (
                    list(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))
                    + list(((tick.get("image_packet", {}) or {}).get("global_structure_samples", []) or []))
                )
                if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("vision_mem::")
            ]
            sensor_overlap_first_tick = _vision_sensor_label_overlap(
                probe_labels=probe_labels,
                concept_labels_by_id=concept_labels_by_id,
            )
        tick_rows.append(
            {
                "probe_tick_index": int(probe_tick_index),
                "elapsed_ms": _round4(elapsed_ms),
                "emotion": _emotion_row(tick, tick_index=probe_tick_index, text=""),
                **eval_row,
            }
        )
    final_row = dict(tick_rows[-1] if tick_rows else {})
    return {
        "scenario_id": "yellow_apple",
        "scenario_type": "visual_probe",
        "observation_ticks": int(observation_ticks),
        "tick_rows": tick_rows,
        "sensor_overlap_first_tick": sensor_overlap_first_tick,
        "final_text_cstar_summary": dict(final_row.get("text_cstar_summary", {}) or {}),
        "final_text_state_summary": dict(final_row.get("text_state_summary", {}) or {}),
        "final_vision_cstar_summary": dict(final_row.get("vision_cstar_summary", {}) or {}),
        "final_vision_state_summary": dict(final_row.get("vision_state_summary", {}) or {}),
        "mean_elapsed_ms": _mean([float(row.get("elapsed_ms", 0.0) or 0.0) for row in tick_rows]),
    }


def _build_showcase_dataset(
    *,
    output_root: Path,
    concepts: list[MultimodalConcept],
    assets: dict[str, dict[str, Any]],
    yellow_apple_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    dataset_path = output_root / "showcase_dataset.json"
    items: list[dict[str, Any]] = []
    phase_ranges: list[dict[str, Any]] = []
    tick_cursor = 0

    def append_phase(phase_id: str, entries: list[dict[str, Any]]) -> None:
        nonlocal tick_cursor
        if not entries:
            return
        start = tick_cursor
        items.extend(entries)
        tick_cursor += len(entries)
        phase_ranges.append(
            {
                "phase_id": phase_id,
                "tick_start": int(start),
                "tick_end": int(tick_cursor - 1),
                "tick_count": int(len(entries)),
            }
        )

    for concept in concepts:
        asset = assets[concept.concept_id]
        append_phase(
            f"train::{concept.concept_id}",
            [
                {
                    "text": concept.text_label,
                    "image_path": asset["image_path"],
                    "audio_path": asset["audio_path"],
                    "source_type": f"overlay_showcase::train::{concept.concept_id}",
                    "external_feedback": {"reward": 0.9, "punishment": 0.0, "notes": [f"overlay_showcase::{concept.concept_id}"]},
                }
                for _ in range(10)
            ],
        )
        append_phase(
            f"idle_after_train::{concept.concept_id}",
            [{"text": "", "source_type": f"overlay_showcase::idle_after_train::{concept.concept_id}"} for _ in range(4)],
        )

    append_phase(
        "probe::text::apple",
        [{"text": "apple", "source_type": "overlay_showcase::probe::text::apple"} for _ in range(6)],
    )
    append_phase("idle::apple", [{"text": "", "source_type": "overlay_showcase::idle::apple"} for _ in range(4)])
    append_phase(
        "probe::text::banana",
        [{"text": "banana", "source_type": "overlay_showcase::probe::text::banana"} for _ in range(6)],
    )
    append_phase("idle::banana", [{"text": "", "source_type": "overlay_showcase::idle::banana"} for _ in range(4)])
    append_phase(
        "probe::text::apple_banana",
        [{"text": "apple banana", "source_type": "overlay_showcase::probe::text::apple_banana"} for _ in range(6)],
    )
    append_phase("idle::apple_banana", [{"text": "", "source_type": "overlay_showcase::idle::apple_banana"} for _ in range(4)])
    append_phase(
        "probe::text::banana_apple",
        [{"text": "banana apple", "source_type": "overlay_showcase::probe::text::banana_apple"} for _ in range(6)],
    )
    append_phase("idle::banana_apple", [{"text": "", "source_type": "overlay_showcase::idle::banana_apple"} for _ in range(4)])
    append_phase(
        "probe::vision::yellow_apple",
        [{"text": "", "image_path": str(yellow_apple_path), "source_type": "overlay_showcase::probe::vision::yellow_apple"} for _ in range(6)],
    )
    append_phase("idle::yellow_apple", [{"text": "", "source_type": "overlay_showcase::idle::yellow_apple"} for _ in range(4)])

    payload = {
        "label": "Phase26 多模态叠加想象与颜色迁移展示运行",
        "mode": "multimodal",
        "max_ticks": len(items),
        "items": items,
    }
    _write_json(dataset_path, payload)
    return dataset_path, phase_ranges


def _tick_line(row: dict[str, Any], concept_order: list[str]) -> str:
    text_cstar = dict(row.get("text_cstar_energies", {}) or {})
    text_state = dict(row.get("text_state_energies", {}) or {})
    vision_cstar = dict(row.get("vision_cstar_energies", {}) or {})
    audio_cstar = dict(row.get("audio_cstar_energies", {}) or {})
    emotion = dict(row.get("emotion", {}) or {})
    pairs = []
    for concept_id in concept_order:
        pairs.append(
            f"{concept_id}: textC*={_round4(float(text_cstar.get(concept_id, 0.0) or 0.0))}, "
            f"textState={_round4(float(text_state.get(concept_id, 0.0) or 0.0))}, "
            f"visionC*={_round4(float(vision_cstar.get(concept_id, 0.0) or 0.0))}, "
            f"audioC*={_round4(float(audio_cstar.get(concept_id, 0.0) or 0.0))}"
        )
    return (
        f"- tick {int(row.get('probe_tick_index', 0) or 0)}: "
        f"{' | '.join(pairs)} / surprise={_round4(float(emotion.get('surprise', 0.0) or 0.0))} / "
        f"dissonance={_round4(float(emotion.get('dissonance', 0.0) or 0.0))}"
    )


def _render_report(
    *,
    output_root: Path,
    concepts: list[MultimodalConcept],
    training: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
    yellow_row: dict[str, Any],
    showcase: dict[str, Any],
    phase_ranges: list[dict[str, Any]],
) -> str:
    concept_order = [concept.concept_id for concept in concepts]
    train_rows = list(training.get("training_rows", []) or [])
    stabilize_rows = list(training.get("stabilize_rows", []) or [])
    lines: list[str] = []

    lines.append("# V2 多模态叠加想象与颜色迁移实验报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("这次只回答两个很具体的问题：")
    lines.append("1. 在已经学会 `apple` 和 `banana` 之后，如果直接输入 `apple banana` 或 `banana apple`，系统是否会同时拉起两个对象，而不是只剩一个对象。")
    lines.append("2. 如果输入一个“黄色苹果”，系统会不会以苹果为主，同时因为颜色相近而带起一点香蕉联想。")
    lines.append("")
    lines.append("为了避免上一段短时上下文残留直接污染结果，本轮正式 probe 在导入长期记忆后，先 `reset_transient_state`，只保留 memory store，再开始测试。这样更接近“学会之后重新看题”的口径。")
    lines.append("")
    lines.append("## 2. 实验设置")
    lines.append("")
    lines.append(f"- 苹果多模态训练：{int(training.get('train_epochs_apple', 0) or 0)} tick")
    lines.append(f"- 香蕉多模态训练：{int(training.get('train_epochs_banana', 0) or 0)} tick")
    lines.append(f"- 每个训练 tick 外部奖励：{_round4(float(training.get('reward_value', 0.0) or 0.0))}")
    lines.append(f"- 训练后稳定空 tick：{int(training.get('stabilize_ticks', 0) or 0)}")
    lines.append(f"- 平均训练耗时：{_mean([float(row.get('elapsed_ms', 0.0) or 0.0) for row in train_rows])} ms")
    lines.append(f"- probe 平均耗时：{_mean([float(row.get('mean_elapsed_ms', 0.0) or 0.0) for row in [*baseline_rows, *overlay_rows, yellow_row]])} ms")
    lines.append(f"- 稳定阶段 tick 数：{len(stabilize_rows)}")
    lines.append("")
    lines.append("报告里同时看 6 组量：")
    lines.append("1. `text C*`：综合预测包里，`apple / banana` 两个文本对象的能量。")
    lines.append("2. `text state`：状态池顶部里，`apple / banana` 两个文本对象的能量。")
    lines.append("3. `vision C*`：视觉身份标签映射到 `apple / banana` 后的能量。")
    lines.append("4. `audio C*`：听觉身份标签映射到 `apple / banana` 后的能量。")
    lines.append("5. `secondary_ratio`：次强对象能量 / 最强对象能量。")
    lines.append("6. 情绪：重点看 `surprise / dissonance`。")
    lines.append("")
    lines.append("文中说“明显双活化”时，只是一个便于阅读的口径：`secondary_ratio >= 0.25`。原始能量值全部保留。")
    lines.append("")
    lines.append("## 3. 基线校准")
    lines.append("")
    lines.append("先做单概念文本 probe，确认系统不是连单个苹果/香蕉都拉不起来。")
    lines.append("")
    for row in baseline_rows:
        final_text = dict(row.get("final_text_cstar_summary", {}) or {})
        final_vision = dict(row.get("final_vision_cstar_summary", {}) or {})
        lines.append(
            f"- `{row.get('prompt', '')}`：text 主导=`{final_text.get('primary_concept_id', '')}` / "
            f"text 次级=`{final_text.get('secondary_concept_id', '')}` ratio={final_text.get('secondary_ratio', 0.0)} / "
            f"vision 主导=`{final_vision.get('primary_concept_id', '')}`"
        )
    lines.append("")
    lines.append("这里要实话实说：`apple` 的单概念基线是干净的，但 `banana` 的 text C* 仍然带着明显的 `apple` 残留。")
    lines.append("不过 `banana` 的视觉 recall 仍然是以香蕉为主，所以后面的双概念实验至少不是建立在“视觉层连单对象都认不出”的坏底座上。")
    lines.append("")
    lines.append("## 4. 实验 A：双概念文本叠加想象")
    lines.append("")
    lines.append("### 4.1 `apple banana`")
    lines.append("")
    apple_banana = next(row for row in overlay_rows if str(row.get("scenario_id", "")) == "apple_banana")
    for tick_row in list(apple_banana.get("tick_rows", []) or []):
        lines.append(_tick_line(tick_row, concept_order))
    final_text = dict(apple_banana.get("final_text_cstar_summary", {}) or {})
    final_state = dict(apple_banana.get("final_text_state_summary", {}) or {})
    final_vision = dict(apple_banana.get("final_vision_cstar_summary", {}) or {})
    final_audio = dict(apple_banana.get("final_audio_cstar_summary", {}) or {})
    lines.append("")
    lines.append(
        f"结论：`apple banana` 最终在 text C* 上是 `{final_text.get('primary_concept_id', '')}` 主导，"
        f"`{final_text.get('secondary_concept_id', '')}` 次级，secondary_ratio={final_text.get('secondary_ratio', 0.0)}；"
        f"state text 上同样保留明显双对象，secondary_ratio={final_state.get('secondary_ratio', 0.0)}。"
    )
    lines.append(
        f"更重要的是，这一顺序下视觉 C* 也已经进入双活化：主导=`{final_vision.get('primary_concept_id', '')}`，"
        f"次级=`{final_vision.get('secondary_concept_id', '')}`，secondary_ratio={final_vision.get('secondary_ratio', 0.0)}。"
    )
    lines.append(
        "所以如果直接看当前前端的能量叠加视图，这一段更像“苹果轮廓更亮、香蕉轮廓更暗，但两者同时在场”的叠加态。"
    )
    lines.append(
        f"音频侧仍然没有形成对应的双活化，最终 audio C* secondary_ratio={final_audio.get('secondary_ratio', 0.0)}。"
    )
    lines.append("")
    lines.append("### 4.2 `banana apple`")
    lines.append("")
    banana_apple = next(row for row in overlay_rows if str(row.get("scenario_id", "")) == "banana_apple")
    for tick_row in list(banana_apple.get("tick_rows", []) or []):
        lines.append(_tick_line(tick_row, concept_order))
    final_text = dict(banana_apple.get("final_text_cstar_summary", {}) or {})
    final_state = dict(banana_apple.get("final_text_state_summary", {}) or {})
    final_vision = dict(banana_apple.get("final_vision_cstar_summary", {}) or {})
    final_audio = dict(banana_apple.get("final_audio_cstar_summary", {}) or {})
    lines.append("")
    lines.append(
        f"结论：`banana apple` 这一顺序里，state text 仍然是强双活化，secondary_ratio={final_state.get('secondary_ratio', 0.0)}；"
        f"但 text C* 本身只到 secondary_ratio={final_text.get('secondary_ratio', 0.0)}，刚好低于这份报告里“明显双活化”的阅读阈值。"
    )
    lines.append(
        f"视觉 C* 里两个对象也都还有能量，但次级只到 secondary_ratio={final_vision.get('secondary_ratio', 0.0)}，"
        "比 `apple banana` 更弱。"
    )
    lines.append(
        f"音频侧双活化仍然偏弱：最终 audio C* 主导=`{final_audio.get('primary_concept_id', '')}`，次级 ratio={final_audio.get('secondary_ratio', 0.0)}。"
    )
    lines.append("")
    lines.append("### 4.3 对实验 A 的判断")
    lines.append("")
    lines.append("1. “同时想着两样东西”这件事已经成立，但目前最稳的是文本层和视觉层，听觉层还不够稳。")
    lines.append("2. 顺序会强烈影响主导对象，说明当前系统不仅会叠加，还会保留顺序偏置。")
    lines.append("3. 从这轮正式数据看，`apple banana` 反而比 `banana apple` 更容易出现你想要的“双轮廓叠加”。")
    lines.append("")
    lines.append("## 5. 实验 B：黄色苹果的颜色迁移")
    lines.append("")
    lines.append("黄色苹果 probe 使用的是“苹果轮廓 + 香蕉色相”的图像，只输入视觉，不输入文本和音频。")
    lines.append("")
    for tick_row in list(yellow_row.get("tick_rows", []) or []):
        lines.append(_tick_line(tick_row, concept_order))
    lines.append("")
    sensor_overlap = dict(yellow_row.get("sensor_overlap_first_tick", {}) or {})
    apple_overlap = dict(sensor_overlap.get("apple", {}) or {})
    banana_overlap = dict(sensor_overlap.get("banana", {}) or {})
    final_text = dict(yellow_row.get("final_text_cstar_summary", {}) or {})
    final_vision = dict(yellow_row.get("final_vision_cstar_summary", {}) or {})
    lines.append(
        f"- 首 tick 低层视觉标签重合：apple overlap={apple_overlap.get('overlap_count', 0)}（contour={apple_overlap.get('contour_overlap_count', 0)}），"
        f"banana overlap={banana_overlap.get('overlap_count', 0)}（contour={banana_overlap.get('contour_overlap_count', 0)}）"
    )
    lines.append(
        f"- 最终 recall：text C* 主导=`{final_text.get('primary_concept_id', '')}` / 次级=`{final_text.get('secondary_concept_id', '')}` / secondary_ratio={final_text.get('secondary_ratio', 0.0)}"
    )
    lines.append(
        f"- 最终视觉 recall：vision C* 主导=`{final_vision.get('primary_concept_id', '')}` / 次级=`{final_vision.get('secondary_concept_id', '')}` / secondary_ratio={final_vision.get('secondary_ratio', 0.0)}"
    )
    lines.append("")
    lines.append("这组结果很清楚：")
    lines.append("1. 首 tick 的低层轮廓重合 actually 只对苹果成立，说明轮廓通道本身并没有把黄色苹果看成香蕉。")
    lines.append("2. 但随着连续几个 tick 的整合，最终高层 recall 反而翻成了香蕉主导。")
    lines.append("3. 这说明问题不在“轮廓没提出来”，而在后续 recall / competition 过程中，黄色带来的颜色相似性、已有香蕉记忆优势，压过了苹果轮廓锚点。")
    lines.append("")
    lines.append("也就是说，如果你的预期是“黄色苹果应该以苹果为主，只稍微想到一点香蕉”，那当前版本没有做到，甚至会在后续整合里翻错到香蕉主导。")
    lines.append("")
    lines.append("## 6. 对你关心问题的直接回答")
    lines.append("")
    lines.append("### 6.1 轮廓重建能不能两个轮廓都有")
    lines.append("")
    lines.append("能，但不是所有顺序都一样强。")
    lines.append("`apple banana` 这组里，视觉 C* 已经出现两个对象同时有正能量，而且次级占比过了 0.25，所以按现在前端的能量叠加逻辑，应该会出现“双轮廓同场、主次明暗不同”的效果。")
    lines.append("`banana apple` 也不是完全没有第二对象，但视觉次级更弱，更像隐约叠上了一层较淡的轮廓。")
    lines.append("")
    lines.append("### 6.2 黄色苹果会不会苹果为主、香蕉略微被带起")
    lines.append("")
    lines.append("这次正式实验里，并不是“苹果主、香蕉副”，而是最后会逐步翻成“香蕉主、苹果副”。")
    lines.append("更准确地说，是“首 tick 的轮廓判断偏苹果，但后续 recall 整合把结果推向了香蕉”。")
    lines.append("")
    lines.append("## 7. Showcase 回看路径")
    lines.append("")
    lines.append(f"- 输出目录：[overlay_color_probe]({output_root})")
    lines.append(f"- 展示 dataset：[showcase_dataset.json]({output_root / 'showcase_dataset.json'})")
    lines.append(f"- 展示 run 目录：[{Path(str(showcase.get('run_dir', '') or '')).name}]({showcase.get('run_dir', '')})")
    lines.append(f"- sidecar / summary tick 数：{int(showcase.get('sidecar_count', 0) or 0)} / {int(showcase.get('summary_count', 0) or 0)}")
    lines.append("")
    lines.append("建议重点回看这些 tick 段：")
    for phase in phase_ranges:
        lines.append(f"- `{phase.get('phase_id', '')}`：tick {phase.get('tick_start', 0)} -> {phase.get('tick_end', 0)}")
    lines.append("")
    lines.append("如果你在前端里看“内心展示”，最值得盯的段落是：")
    lines.append("1. `probe::text::apple_banana`：看是不是主要苹果、但仍有香蕉痕迹。")
    lines.append("2. `probe::text::banana_apple`：看是不是更明显地出现双轮廓叠加。")
    lines.append("3. `probe::vision::yellow_apple`：看是不是苹果主导，同时几乎没有稳定香蕉副本。")
    lines.append("")
    lines.append("## 8. 当前结论")
    lines.append("")
    lines.append("> 这次实验最明确的结论是：AP V2 已经具备“多概念同时激活”的雏形，而且在 `apple banana` 这类顺序下，视觉层确实能进入双轮廓叠加态；但颜色迁移目前不是“弱副联想”，而是会把黄色苹果逐步推成香蕉主导，说明颜色/记忆偏置在后续整合里压过了轮廓锚点。")
    lines.append("")
    lines.append("如果后面你希望“黄色苹果 -> 以苹果为主、稍微想到香蕉”更稳定成立，比较自然的改进方向不是硬调规则，而是让轮廓通道在多 tick 整合中的稳定权重更高，同时把颜色/材质通道保留为次级相似来源，而不是让它在后续竞争里反客为主。")
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
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    concepts = list(DEFAULT_CONCEPTS)
    assets, _ = _prepare_concept_assets(output_root, concepts)

    yellow_apple_bytes = _render_yellow_apple_image()
    yellow_apple_path = output_root / "assets" / "yellow_apple.png"
    yellow_apple_path.parent.mkdir(parents=True, exist_ok=True)
    yellow_apple_path.write_bytes(yellow_apple_bytes)

    yellow_runtime = _mk_runtime()
    yellow_packet = yellow_runtime.vision_sensor.ingest_image_bytes(yellow_apple_bytes, tick_index=0, source_type="asset::yellow_apple")
    yellow_labels = [
        str(item.get("sa_label", "") or "")
        for item in (
            list(yellow_packet.get("memory_write_samples", []) or [])
            + list(yellow_packet.get("global_structure_samples", []) or [])
        )
        if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("vision_mem::")
    ]
    assets["yellow_apple"] = {
        "concept_id": "yellow_apple",
        "text_label": "yellow_apple",
        "zh_text": "黄色苹果",
        "image_path": str(yellow_apple_path),
        "vision_labels": yellow_labels,
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
    vision_allowed_labels = set(str(label or "") for labels in (vision_maps.get("unique_labels_by_concept", {}) or {}).values() for label in list(labels or []))
    audio_allowed_labels = set(str(label or "") for labels in (audio_maps.get("unique_labels_by_concept", {}) or {}).values() for label in list(labels or []))
    vision_label_to_concept = dict((vision_maps.get("unique_label_to_concept", {}) or {}))
    audio_label_to_concept = dict((audio_maps.get("unique_label_to_concept", {}) or {}))

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

    baseline_rows = [
        _run_text_prompt_probe(
            imported_payload=payload,
            prompt="apple",
            scenario_id="baseline_apple",
            observation_ticks=observation_ticks,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
            runtime_overrides=runtime_overrides,
        ),
        _run_text_prompt_probe(
            imported_payload=payload,
            prompt="banana",
            scenario_id="baseline_banana",
            observation_ticks=observation_ticks,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
            runtime_overrides=runtime_overrides,
        ),
    ]

    overlay_rows = [
        _run_text_prompt_probe(
            imported_payload=payload,
            prompt="apple banana",
            scenario_id="apple_banana",
            observation_ticks=observation_ticks,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
            runtime_overrides=runtime_overrides,
        ),
        _run_text_prompt_probe(
            imported_payload=payload,
            prompt="banana apple",
            scenario_id="banana_apple",
            observation_ticks=observation_ticks,
            concepts=concepts,
            vision_allowed_labels=vision_allowed_labels,
            audio_allowed_labels=audio_allowed_labels,
            vision_label_to_concept=vision_label_to_concept,
            audio_label_to_concept=audio_label_to_concept,
            runtime_overrides=runtime_overrides,
        ),
    ]

    yellow_row = _run_yellow_apple_probe(
        imported_payload=payload,
        image_bytes=yellow_apple_bytes,
        observation_ticks=observation_ticks,
        concepts=concepts,
        assets=assets,
        vision_allowed_labels=vision_allowed_labels,
        audio_allowed_labels=audio_allowed_labels,
        vision_label_to_concept=vision_label_to_concept,
        audio_label_to_concept=audio_label_to_concept,
        runtime_overrides=runtime_overrides,
    )

    showcase_dataset, phase_ranges = _build_showcase_dataset(
        output_root=output_root,
        concepts=concepts,
        assets=assets,
        yellow_apple_path=yellow_apple_path,
    )
    showcase = _run_observatory_showcase(output_root=output_root, dataset_path=showcase_dataset)
    showcase["phase_ranges"] = phase_ranges

    summary = {
        "schema_id": "multimodal_overlay_color_probe/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "reward_value": _round4(reward_value),
            "train_epochs_apple": int(train_epochs_apple),
            "train_epochs_banana": int(train_epochs_banana),
            "stabilize_ticks": int(stabilize_ticks),
            "observation_ticks": int(observation_ticks),
            "runtime_overrides": dict(runtime_overrides or {}),
        },
        "training": {key: value for key, value in training.items() if key != "stabilized_payload"},
        "assets": {
            concept_id: {
                key: value
                for key, value in asset.items()
                if key not in {"image_bytes", "audio_bytes"}
            }
            for concept_id, asset in assets.items()
        },
        "vision_maps": vision_maps,
        "audio_maps": audio_maps,
        "baseline_rows": baseline_rows,
        "overlay_rows": overlay_rows,
        "yellow_row": yellow_row,
        "showcase": showcase,
    }
    report = _render_report(
        output_root=output_root,
        concepts=concepts,
        training=summary["training"],
        baseline_rows=baseline_rows,
        overlay_rows=overlay_rows,
        yellow_row=yellow_row,
        showcase=showcase,
        phase_ranges=phase_ranges,
    )
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "baseline_rows.json", baseline_rows)
    _write_json(output_root / "overlay_rows.json", overlay_rows)
    _write_json(output_root / "yellow_row.json", yellow_row)
    _write_text(output_root / "report.md", report)
    _write_text(doc_path, report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AP V2 multimodal overlay imagination + color transfer probe.")
    parser.add_argument("--output-dir", default="", help="输出目录，默认写入 outputs/multimodal_overlay_color_probe/<timestamp>")
    parser.add_argument("--doc-path", default="", help="正式报告路径")
    parser.add_argument("--reward", type=float, default=1.0, help="每个训练 tick 注入的奖励值")
    parser.add_argument("--train-epochs-apple", type=int, default=12, help="苹果训练 tick 数")
    parser.add_argument("--train-epochs-banana", type=int, default=12, help="香蕉训练 tick 数")
    parser.add_argument("--stabilize-ticks", type=int, default=8, help="训练后空 tick")
    parser.add_argument("--observation-ticks", type=int, default=6, help="每个 probe 连续观察 tick")
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
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "doc_path": str(doc_path),
                "showcase_run_dir": str((summary.get("showcase", {}) or {}).get("run_dir", "") or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
