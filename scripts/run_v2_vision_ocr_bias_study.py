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

from PIL import Image, ImageDraw, ImageFilter, ImageFont

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v2_vision_ocr_probe import (  # noqa: E402
    FONT_CANDIDATES,
    OCRPair,
    _build_runtime,
    _evaluate_probe,
    _inject_reward,
    _probe_overrides,
    _round4,
    _run_multimodal_tick,
    _training_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vision_ocr_bias_study"


@dataclass(frozen=True)
class VariantSpec:
    pair_id: str
    glyph: str
    text_label: str
    rotate_deg: float
    font_index: int
    offset_x: int
    offset_y: int
    font_size: int
    stroke_width: int
    fill_value: int
    blur_radius: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _load_font(font_index: int, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [path for path in FONT_CANDIDATES if path.exists()]
    if candidates:
        chosen = candidates[int(font_index) % len(candidates)]
        try:
            return ImageFont.truetype(str(chosen), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _render_variant(spec: VariantSpec, *, size: tuple[int, int] = (256, 128)) -> bytes:
    canvas = Image.new("RGB", size, color=(12, 12, 12))
    glyph_layer = Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(glyph_layer)
    font = _load_font(spec.font_index, spec.font_size)
    bbox = draw.textbbox((0, 0), spec.glyph, font=font, stroke_width=spec.stroke_width)
    text_w = max(1, int(bbox[2] - bbox[0]))
    text_h = max(1, int(bbox[3] - bbox[1]))
    x = int((size[0] - text_w) / 2 - bbox[0] + spec.offset_x)
    y = int((size[1] - text_h) / 2 - bbox[1] + spec.offset_y)
    fill = int(max(180, min(255, spec.fill_value)))
    draw.text(
        (x, y),
        spec.glyph,
        font=font,
        fill=(fill, fill, fill, 255),
        stroke_width=spec.stroke_width,
        stroke_fill=(max(0, fill - 18), max(0, fill - 18), max(0, fill - 18), 255),
    )
    rotated = glyph_layer.rotate(spec.rotate_deg, resample=Image.Resampling.BICUBIC, expand=False)
    if spec.blur_radius > 0.0:
        rotated = rotated.filter(ImageFilter.GaussianBlur(radius=spec.blur_radius))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), rotated).convert("RGB")
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _build_variant_sets() -> tuple[list[VariantSpec], list[VariantSpec]]:
    train_specs: list[VariantSpec] = []
    test_specs: list[VariantSpec] = []
    train_recipes = [
        (-10.0, 0, -8, -2, 104, 3, 248, 0.0),
        (-6.0, 1, -3, 4, 102, 3, 242, 0.0),
        (-2.0, 2, 1, -4, 100, 2, 236, 0.35),
        (3.0, 0, 5, 2, 98, 3, 250, 0.0),
        (7.0, 1, -4, 5, 106, 3, 244, 0.45),
        (11.0, 2, 7, -3, 96, 2, 232, 0.25),
    ]
    test_recipes = [
        (-8.0, 1, 6, -1, 103, 3, 246, 0.2),
        (-1.0, 0, -6, 3, 97, 2, 234, 0.5),
        (5.0, 2, 3, -5, 108, 3, 252, 0.0),
        (9.0, 1, -2, 6, 101, 2, 238, 0.3),
    ]
    glyphs = [("digit_3", "3", "three"), ("digit_8", "8", "eight")]
    for prefix, glyph, text_label in glyphs:
        for idx, recipe in enumerate(train_recipes):
            rotate_deg, font_index, offset_x, offset_y, font_size, stroke_width, fill_value, blur_radius = recipe
            train_specs.append(
                VariantSpec(
                    pair_id=f"{prefix}_train_{idx:02d}",
                    glyph=glyph,
                    text_label=text_label,
                    rotate_deg=rotate_deg,
                    font_index=font_index,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    fill_value=fill_value,
                    blur_radius=blur_radius,
                )
            )
        for idx, recipe in enumerate(test_recipes):
            rotate_deg, font_index, offset_x, offset_y, font_size, stroke_width, fill_value, blur_radius = recipe
            test_specs.append(
                VariantSpec(
                    pair_id=f"{prefix}_test_{idx:02d}",
                    glyph=glyph,
                    text_label=text_label,
                    rotate_deg=rotate_deg,
                    font_index=font_index,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    fill_value=fill_value,
                    blur_radius=blur_radius,
                )
            )
    return train_specs, test_specs


def _render_image_map(specs: list[VariantSpec], *, output_dir: Path) -> dict[str, bytes]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, bytes] = {}
    for spec in specs:
        raw = _render_variant(spec)
        image_map[spec.pair_id] = raw
        (output_dir / f"{spec.pair_id}.png").write_bytes(raw)
    return image_map


def _filter_specs(
    specs: list[VariantSpec],
    *,
    per_label_limit: int | None = None,
    allowed_labels: list[str] | None = None,
) -> list[VariantSpec]:
    allowed = {str(item or "") for item in (allowed_labels or []) if str(item or "")}
    grouped: dict[str, list[VariantSpec]] = {}
    for spec in specs:
        if allowed and str(spec.text_label) not in allowed:
            continue
        grouped.setdefault(str(spec.text_label), []).append(spec)
    limited: list[VariantSpec] = []
    for label in sorted(grouped.keys()):
        rows = grouped[label]
        if per_label_limit is not None:
            rows = rows[: max(1, int(per_label_limit))]
        limited.extend(rows)
    return limited


def _group_specs_by_label(specs: list[VariantSpec]) -> dict[str, list[VariantSpec]]:
    grouped: dict[str, list[VariantSpec]] = {}
    for spec in specs:
        grouped.setdefault(str(spec.text_label), []).append(spec)
    return {key: grouped[key] for key in sorted(grouped.keys())}


def _run_idle_ticks(
    runtime: Any,
    *,
    tick_index: int,
    count: int,
    source_type: str,
    meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    next_tick = int(tick_index)
    for _ in range(max(0, int(count))):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=next_tick,
            text="",
            image_bytes=None,
            source_type=source_type,
        )
        row = {
            "tick_index": next_tick,
            "elapsed_ms": _round4(elapsed_ms),
            "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
            "top_preview": [str(item.get("display", "") or item.get("display_text", "") or "") for item in ((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])[:6]],
        }
        if meta:
            row.update(dict(meta))
        rows.append(row)
        next_tick += 1
    return rows, next_tick


def _train_large_variant_memory(
    *,
    train_specs: list[VariantSpec],
    train_image_map: dict[str, bytes],
    train_epochs: int,
    train_raw_budget: int,
    train_patch_budget: int,
    train_focus_budget: int,
    stabilize_ticks: int,
    reward_value: float,
    reset_between_label_blocks: bool = False,
    label_block_stabilize_ticks: int = 0,
    training_overrides_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = _training_overrides(
        raw_budget=train_raw_budget,
        patch_budget=train_patch_budget,
        focus_budget=train_focus_budget,
    )
    overrides.update(dict(training_overrides_extra or {}))
    runtime = _build_runtime(
        overrides=overrides
    )
    tick_index = 0
    training_rows: list[dict[str, Any]] = []
    label_block_rows: list[dict[str, Any]] = []
    grouped_specs = _group_specs_by_label(train_specs)
    label_order = list(grouped_specs.keys())
    for epoch in range(int(train_epochs)):
        for label_index, text_label in enumerate(label_order):
            for spec in grouped_specs[text_label]:
                pair = OCRPair(pair_id=spec.pair_id, glyph=spec.glyph, text_label=spec.text_label, rotate_deg=spec.rotate_deg)
                tick, elapsed_ms = _run_multimodal_tick(
                    runtime,
                    tick_index=tick_index,
                    text=spec.text_label,
                    image_bytes=train_image_map[spec.pair_id],
                    source_type=f"vision_ocr_bias_train::{spec.pair_id}",
                )
                reward_payload = _inject_reward(runtime, tick_index=tick_index, tick=tick, pair=pair, reward=reward_value)
                training_rows.append(
                    {
                        "tick_index": tick_index,
                        "epoch": epoch,
                        "phase": "train",
                        "pair_id": spec.pair_id,
                        "text_label": spec.text_label,
                        "elapsed_ms": _round4(elapsed_ms),
                        "raw_sample_count": int((tick.get("image_packet", {}) or {}).get("total_patch_count", 0) or 0),
                        "memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))),
                        "focus_memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("focus_priority_samples", []) or []))),
                        "reward_injected": _round4(float(reward_payload.get("reward", 0.0) or 0.0)),
                        "memory_count_after_tick": int(tick.get("memory_count", 0) or 0),
                    }
                )
                tick_index += 1
            if label_block_stabilize_ticks > 0:
                rows, tick_index = _run_idle_ticks(
                    runtime,
                    tick_index=tick_index,
                    count=label_block_stabilize_ticks,
                    source_type=f"vision_ocr_label_stabilize::{text_label}",
                    meta={"epoch": epoch, "phase": "label_block_stabilize", "text_label": text_label},
                )
                label_block_rows.extend(rows)
            should_reset = bool(reset_between_label_blocks) and (label_index < len(label_order) - 1 or epoch < int(train_epochs) - 1)
            if should_reset:
                runtime.reset_transient_state()
                label_block_rows.append(
                    {
                        "tick_index": tick_index,
                        "epoch": epoch,
                        "phase": "label_block_reset",
                        "text_label": text_label,
                        "memory_count_after_reset": int(runtime.memory_store.count() or 0),
                    }
                )

    stabilize_rows, tick_index = _run_idle_ticks(
        runtime,
        tick_index=tick_index,
        count=stabilize_ticks,
        source_type="vision_ocr_bias_stabilize",
        meta={"phase": "final_stabilize"},
    )
    payload = runtime.export_payload()
    return {
        "payload": payload,
        "training_rows": training_rows,
        "label_block_rows": label_block_rows,
        "stabilize_rows": stabilize_rows,
        "train_epochs": int(train_epochs),
        "train_raw_budget": int(train_raw_budget),
        "train_patch_budget": int(train_patch_budget),
        "train_focus_budget": int(train_focus_budget),
    }


def _probe_variant(
    *,
    payload: dict[str, Any],
    spec: VariantSpec,
    image_bytes: bytes,
    distractor_texts: list[str],
    raw_budget: int,
    patch_budget: int,
    focus_budget: int,
    observation_ticks: int,
    probe_overrides_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = _probe_overrides(raw_budget=raw_budget, patch_budget=patch_budget, focus_budget=focus_budget)
    overrides.update(dict(probe_overrides_extra or {}))
    runtime = _build_runtime(
        overrides=overrides
    )
    runtime.import_payload({"memory_store": copy.deepcopy(payload.get("memory_store", {}))})
    tick_rows: list[dict[str, Any]] = []
    final_tick: dict[str, Any] = {}
    for probe_index in range(int(observation_ticks)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=probe_index,
            text="",
            image_bytes=image_bytes,
            source_type=f"vision_ocr_bias_probe::{spec.pair_id}",
        )
        final_tick = tick
        eval_row = _evaluate_probe(
            tick=tick,
            target_text=spec.text_label,
            distractor_texts=distractor_texts,
        )
        tick_rows.append(
            {
                "probe_tick_index": probe_index,
                "elapsed_ms": _round4(elapsed_ms),
                "strict_success": bool(eval_row["strict_success"]),
                "bn_best_text": str(eval_row["bn_best_text"]),
                "bn_target_rank": int(eval_row["bn_target_rank"]),
                "cstar_best_text": str(eval_row["cstar_best_text"]),
                "cstar_margin": _round4(eval_row["cstar_margin"]),
                "raw_sample_count": int((tick.get("image_packet", {}) or {}).get("total_patch_count", 0) or 0),
                "memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("memory_write_samples", []) or []))),
                "focus_memory_write_count": int(len(((tick.get("image_packet", {}) or {}).get("focus_priority_samples", []) or []))),
            }
        )
    final_eval = _evaluate_probe(
        tick=final_tick,
        target_text=spec.text_label,
        distractor_texts=distractor_texts,
    )
    return {
        "pair_id": spec.pair_id,
        "target_text_label": spec.text_label,
        "raw_budget": int(raw_budget),
        "patch_budget": int(patch_budget),
        "focus_budget": int(focus_budget),
        "observation_ticks": int(observation_ticks),
        "mean_elapsed_ms": _round4(sum(float(row["elapsed_ms"]) for row in tick_rows) / max(1, len(tick_rows))),
        "tick_rows": tick_rows,
        "final_evaluation": final_eval,
    }


def _budget_modes(raw_budget: int) -> list[dict[str, Any]]:
    scaled_patch = max(8, min(64, int(round(raw_budget / 32.0))))
    scaled_focus = max(4, min(scaled_patch, int(round(scaled_patch / 2.0))))
    return [
        {"mode": "scaled", "raw_budget": int(raw_budget), "patch_budget": int(scaled_patch), "focus_budget": int(scaled_focus)},
        {"mode": "capped16", "raw_budget": int(raw_budget), "patch_budget": 16, "focus_budget": 8},
    ]


def _summarize_condition(*, mode: str, raw_budget: int, observation_ticks: int, probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_hits = 0
    by_label: dict[str, dict[str, int]] = {}
    mean_elapsed_ms = 0.0
    for row in probe_rows:
        final_eval = dict(row.get("final_evaluation", {}) or {})
        target = str(row.get("target_text_label", "") or "")
        predicted = str(final_eval.get("cstar_best_text", "") or final_eval.get("bn_best_text", "") or "")
        strict_success = bool(final_eval.get("strict_success", False))
        strict_hits += 1 if strict_success else 0
        label_stats = by_label.setdefault(target, {"total": 0, "strict_success": 0, "predicted_as_target": 0, "predicted_as_other": 0})
        label_stats["total"] += 1
        label_stats["strict_success"] += 1 if strict_success else 0
        if predicted == target:
            label_stats["predicted_as_target"] += 1
        else:
            label_stats["predicted_as_other"] += 1
        mean_elapsed_ms += float(row.get("mean_elapsed_ms", 0.0) or 0.0)
    total = max(1, len(probe_rows))
    return {
        "mode": mode,
        "raw_budget": int(raw_budget),
        "observation_ticks": int(observation_ticks),
        "probe_count": len(probe_rows),
        "strict_accuracy": _round4(strict_hits / total),
        "mean_elapsed_ms": _round4(mean_elapsed_ms / total),
        "by_label": by_label,
        "probe_rows": probe_rows,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# V2 OCR 偏置原因扩大规模分析")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append("这轮实验不是再证明能不能识别，而是专门分析此前出现的：")
    lines.append("")
    lines.append("- 为什么 `digit_3` 更容易被压成 `digit_8`")
    lines.append("- 为什么某些短实验里 `512 raw` 反而看起来比 `1024 raw` 更好")
    lines.append("- 原因更像是 `raw 预算` 问题，还是 `memory write 比例` 问题")
    lines.append("")
    lines.append("## 2. 方法")
    lines.append("")
    lines.append("- 训练集：每类 6 个手写变体")
    lines.append("- 测试集：每类 4 个未见变体")
    lines.append("- 训练预算：1536 raw / 64 patch / 32 focus")
    lines.append("- 训练轮数：8 epochs")
    lines.append("- 稳定空 tick：6")
    lines.append("- 对照预算：512 / 1024 / 1536")
    lines.append("- 对照模式：")
    lines.append("  - `scaled`：patch/focus 随 raw 提高")
    lines.append("  - `capped16`：patch=16, focus=8，固定不变")
    lines.append("")
    lines.append("## 3. 关键结果")
    lines.append("")
    for row in summary.get("conditions", []):
        lines.append(
            f"- `{row['mode']}` / raw=`{row['raw_budget']}` / obs=`{row['observation_ticks']}`: "
            f"strict_accuracy=`{row['strict_accuracy']}` mean_elapsed_ms=`{row['mean_elapsed_ms']}`"
        )
        for label, stats in (row.get("by_label", {}) or {}).items():
            lines.append(
                f"  - `{label}`: success `{stats.get('strict_success', 0)}` / `{stats.get('total', 0)}`, "
                f"pred_as_target `{stats.get('predicted_as_target', 0)}`, pred_as_other `{stats.get('predicted_as_other', 0)}`"
            )
    lines.append("")
    lines.append("## 4. 初步解释")
    lines.append("")
    lines.append("如果 `scaled` 在高 raw 下变差，但 `capped16` 恢复，说明问题更像：")
    lines.append("")
    lines.append("- 不是高 raw 本身坏了")
    lines.append("- 而是 raw 提高时，memory write/focus write 也同步放大")
    lines.append("- 导致更多模糊、通用、背景或共享笔画特征被写入记忆")
    lines.append("- 从而让类别间可分辨性下降")
    lines.append("")
    lines.append("如果两种模式都一起变差，才更像是 raw state 本身带来的状态池噪声增强。")
    lines.append("")
    return "\n".join(lines)


def run_bias_study(
    *,
    output_root: Path,
    train_epochs: int = 8,
    train_raw_budget: int = 1536,
    train_patch_budget: int = 64,
    train_focus_budget: int = 32,
    stabilize_ticks: int = 6,
    reward_value: float = 1.0,
    raw_budgets: list[int] | None = None,
    observation_ticks_list: list[int] | None = None,
    train_spec_limit_per_label: int | None = None,
    test_spec_limit_per_label: int | None = None,
    allowed_labels: list[str] | None = None,
    reset_between_label_blocks: bool = False,
    label_block_stabilize_ticks: int = 0,
    training_overrides_extra: dict[str, Any] | None = None,
    probe_overrides_extra: dict[str, Any] | None = None,
    experiment_tag: str = "",
) -> dict[str, Any]:
    raw_budgets = list(raw_budgets or [512, 1024, 1536])
    observation_ticks_list = list(observation_ticks_list or [2, 4])
    train_specs_all, test_specs_all = _build_variant_sets()
    train_specs = _filter_specs(train_specs_all, per_label_limit=train_spec_limit_per_label, allowed_labels=allowed_labels)
    test_specs = _filter_specs(test_specs_all, per_label_limit=test_spec_limit_per_label, allowed_labels=allowed_labels)
    train_dir = output_root / "train_images"
    test_dir = output_root / "test_images"
    train_image_map = _render_image_map(train_specs, output_dir=train_dir)
    test_image_map = _render_image_map(test_specs, output_dir=test_dir)

    training = _train_large_variant_memory(
        train_specs=train_specs,
        train_image_map=train_image_map,
        train_epochs=train_epochs,
        train_raw_budget=train_raw_budget,
        train_patch_budget=train_patch_budget,
        train_focus_budget=train_focus_budget,
        stabilize_ticks=stabilize_ticks,
        reward_value=reward_value,
        reset_between_label_blocks=reset_between_label_blocks,
        label_block_stabilize_ticks=label_block_stabilize_ticks,
        training_overrides_extra=training_overrides_extra,
    )

    conditions: list[dict[str, Any]] = []
    all_text_labels = sorted({str(spec.text_label) for spec in train_specs + test_specs})
    for raw_budget in raw_budgets:
        for mode_spec in _budget_modes(raw_budget):
            for observation_ticks in observation_ticks_list:
                probe_rows: list[dict[str, Any]] = []
                for spec in test_specs:
                    row = _probe_variant(
                        payload=training["payload"],
                        spec=spec,
                        image_bytes=test_image_map[spec.pair_id],
                        distractor_texts=[text for text in all_text_labels if text != spec.text_label],
                        raw_budget=int(mode_spec["raw_budget"]),
                        patch_budget=int(mode_spec["patch_budget"]),
                        focus_budget=int(mode_spec["focus_budget"]),
                        observation_ticks=int(observation_ticks),
                        probe_overrides_extra=probe_overrides_extra,
                    )
                    probe_rows.append(row)
                conditions.append(
                    _summarize_condition(
                        mode=str(mode_spec["mode"]),
                        raw_budget=int(mode_spec["raw_budget"]),
                        observation_ticks=int(observation_ticks),
                        probe_rows=probe_rows,
                    )
                )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_tag": str(experiment_tag or ""),
        "train_spec_count": len(train_specs),
        "test_spec_count": len(test_specs),
        "training": {
            "train_epochs": int(train_epochs),
            "train_raw_budget": int(train_raw_budget),
            "train_patch_budget": int(train_patch_budget),
            "train_focus_budget": int(train_focus_budget),
            "stabilize_ticks": int(stabilize_ticks),
            "label_block_stabilize_ticks": int(label_block_stabilize_ticks),
            "reset_between_label_blocks": bool(reset_between_label_blocks),
            "reward_value": float(reward_value),
            "train_spec_limit_per_label": None if train_spec_limit_per_label is None else int(train_spec_limit_per_label),
            "test_spec_limit_per_label": None if test_spec_limit_per_label is None else int(test_spec_limit_per_label),
            "allowed_labels": list(allowed_labels or []),
            "training_overrides_extra": dict(training_overrides_extra or {}),
            "training_rows": training["training_rows"],
            "label_block_rows": training["label_block_rows"],
            "stabilize_rows": training["stabilize_rows"],
        },
        "probe_overrides_extra": dict(probe_overrides_extra or {}),
        "conditions": conditions,
    }
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "report.md", _render_report(summary))
    return summary


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = run_bias_study(output_root=output_root)
    print(json.dumps({"output_dir": str(output_root), "condition_count": len(summary.get("conditions", []))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
