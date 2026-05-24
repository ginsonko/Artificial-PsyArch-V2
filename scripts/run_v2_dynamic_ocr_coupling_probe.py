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

from PIL import Image

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v2_vision_ocr_probe import (
    OCRPair,
    _build_runtime,
    _evaluate_probe,
    _inject_reward,
    _probe_overrides,
    _render_handwritten_image,
    _round4,
    _run_multimodal_tick,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "dynamic_ocr_coupling"


@dataclass(frozen=True)
class ProbeCondition:
    name: str
    dynamic_enabled: bool
    auto_reorient_enabled: bool
    execute_selected_actions: bool
    positions: tuple[int, ...]


DEFAULT_PAIRS = [
    OCRPair(pair_id="digit_3", glyph="3", text_label="three", rotate_deg=-6.0),
    OCRPair(pair_id="digit_8", glyph="8", text_label="eight", rotate_deg=5.0),
]

DEFAULT_CONDITIONS = [
    ProbeCondition("static_center_passive", False, False, False, (0, 0, 0, 0, 0, 0)),
    ProbeCondition("moving_passive_no_dynamic", False, False, False, (0, 12, 24, 36, 48, 60)),
    ProbeCondition("moving_dynamic_no_auto", True, False, False, (0, 12, 24, 36, 48, 60)),
    ProbeCondition("moving_dynamic_auto", True, True, False, (0, 12, 24, 36, 48, 60)),
    ProbeCondition("moving_dynamic_full", True, True, True, (0, 12, 24, 36, 48, 60)),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic OCR coupling probe with incremental checkpoints.")
    parser.add_argument("--output-tag", default="", help="Optional suffix for the output directory name.")
    parser.add_argument("--pairs", nargs="*", default=[], help="Subset of pair ids to run, e.g. digit_3 digit_8.")
    parser.add_argument("--conditions", nargs="*", default=[], help="Subset of condition names to run.")
    parser.add_argument("--train-plan", default="8,4,4,2", help="Comma separated staged epochs, e.g. 4,2.")
    parser.add_argument("--probe-raw-budget", type=int, default=64)
    parser.add_argument("--probe-patch-budget", type=int, default=8)
    parser.add_argument("--probe-focus-budget", type=int, default=4)
    parser.add_argument("--positions", default="", help="Override positions for moving conditions, e.g. 0,24,48,72.")
    parser.add_argument("--canvas-width", type=int, default=256)
    parser.add_argument("--canvas-height", type=int, default=128)
    parser.add_argument("--train-raw-budget", type=int, default=1536)
    parser.add_argument("--train-patch-budget", type=int, default=64)
    parser.add_argument("--train-focus-budget", type=int, default=32)
    parser.add_argument("--stabilize-ticks", type=int, default=6)
    parser.add_argument("--dynamic-summary-limit", type=int, default=4)
    parser.add_argument("--probe-read-only", action="store_true", help="Disable long-term memory writes during probe ticks.")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from summary.json under output dir.")
    return parser.parse_args()


def _parse_int_tuple(raw: str, *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    clean = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not clean:
        return tuple(int(item) for item in fallback)
    values: list[int] = []
    for part in clean:
        try:
            values.append(int(part))
        except Exception:
            continue
    return tuple(values) if values else tuple(int(item) for item in fallback)


def _select_pairs(requested_ids: list[str]) -> list[OCRPair]:
    if not requested_ids:
        return list(DEFAULT_PAIRS)
    wanted = {str(item or "").strip() for item in requested_ids if str(item or "").strip()}
    return [pair for pair in DEFAULT_PAIRS if pair.pair_id in wanted]


def _select_conditions(requested_names: list[str], *, positions_override: tuple[int, ...] | None, dynamic_summary_limit: int) -> list[ProbeCondition]:
    wanted = {str(item or "").strip() for item in requested_names if str(item or "").strip()}
    rows: list[ProbeCondition] = []
    for base in DEFAULT_CONDITIONS:
        if wanted and base.name not in wanted:
            continue
        positions = positions_override if (positions_override and "moving" in base.name) else base.positions
        rows.append(
            ProbeCondition(
                name=base.name,
                dynamic_enabled=base.dynamic_enabled if dynamic_summary_limit > 0 else False,
                auto_reorient_enabled=base.auto_reorient_enabled,
                execute_selected_actions=base.execute_selected_actions,
                positions=tuple(int(item) for item in positions),
            )
        )
    return rows


def _shift_image_bytes(raw: bytes, *, dx: int, dy: int = 0, size: tuple[int, int] = (256, 128)) -> bytes:
    base = Image.open(BytesIO(raw)).convert("RGB")
    canvas = Image.new("RGB", size, color=(12, 12, 12))
    canvas.paste(base, (int(dx), int(dy)))
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _probe_runtime(
    *,
    payload: dict[str, Any],
    raw_budget: int,
    patch_budget: int,
    focus_budget: int,
    dynamic_enabled: bool,
    auto_reorient_enabled: bool,
    dynamic_summary_limit: int,
    read_only: bool,
) -> Any:
    runtime = _build_runtime(
        overrides=_probe_overrides(
            raw_budget=raw_budget,
            patch_budget=patch_budget,
            focus_budget=focus_budget,
        )
    )
    runtime.import_payload({"memory_store": copy.deepcopy(payload.get("memory_store", {}))})
    runtime.vision_sensor.dynamic_summary_limit = int(dynamic_summary_limit) if dynamic_enabled else 0
    runtime.config = type(runtime.config)(**{**runtime.config.to_dict(), "vision_auto_surprise_reorient_enabled": bool(auto_reorient_enabled)})
    if read_only:
        runtime.memory_store.write_memory_batch = lambda rows: []
    runtime.vision_sensor.move_gaze(0.5, 0.5)
    return runtime


def _train_runtime(
    *,
    pairs: list[OCRPair],
    image_map: dict[str, bytes],
    train_plan: tuple[int, ...],
    train_raw_budget: int,
    train_patch_budget: int,
    train_focus_budget: int,
    stabilize_ticks: int,
) -> tuple[Any, dict[str, Any]]:
    runtime = _build_runtime(
        overrides={
            **_probe_overrides(raw_budget=train_raw_budget, patch_budget=train_patch_budget, focus_budget=train_focus_budget),
            "intrinsic_feedback_enabled": False,
        }
    )
    tick_index = 0
    trained_epochs = 0
    stages: list[dict[str, Any]] = []
    for segment_epochs in train_plan:
        for _ in range(int(segment_epochs)):
            for pair in pairs:
                tick, elapsed_ms = _run_multimodal_tick(
                    runtime,
                    tick_index=tick_index,
                    text=pair.text_label,
                    image_bytes=image_map[pair.pair_id],
                    source_type=f"dynamic_ocr_train::{pair.pair_id}",
                    execute_selected_actions=True,
                )
                _inject_reward(runtime, tick_index=tick_index, tick=tick, pair=pair, reward=1.0)
                tick_index += 1
        trained_epochs += int(segment_epochs)
        validation_rows: list[dict[str, Any]] = []
        for pair in pairs:
            probe_runtime = _build_runtime(overrides=_probe_overrides(raw_budget=min(1024, train_raw_budget), patch_budget=max(16, min(train_patch_budget, 32)), focus_budget=max(8, min(train_focus_budget, 16))))
            probe_runtime.import_payload({"memory_store": copy.deepcopy(runtime.export_payload().get("memory_store", {}))})
            probe_tick, probe_elapsed_ms = _run_multimodal_tick(
                probe_runtime,
                tick_index=0,
                text="",
                image_bytes=image_map[pair.pair_id],
                source_type=f"dynamic_ocr_train_validate::{pair.pair_id}",
                execute_selected_actions=False,
            )
            final_eval = _evaluate_probe(
                tick=probe_tick,
                target_text=pair.text_label,
                distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
            )
            validation_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "strict_success": bool(final_eval.get("strict_success", False)),
                    "bn_best_text": str(final_eval.get("bn_best_text", "") or ""),
                    "cstar_best_text": str(final_eval.get("cstar_best_text", "") or ""),
                    "cstar_margin": _round4(float(final_eval.get("cstar_margin", 0.0) or 0.0)),
                    "elapsed_ms": _round4(probe_elapsed_ms),
                }
            )
        accepted = all(bool(row.get("strict_success", False)) for row in validation_rows)
        stages.append(
            {
                "trained_epochs": int(trained_epochs),
                "validation_rows": validation_rows,
                "accepted": bool(accepted),
            }
        )
        if accepted:
            break

    for _ in range(max(0, int(stabilize_ticks))):
        _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            image_bytes=None,
            source_type="dynamic_ocr_stabilize",
            execute_selected_actions=False,
        )
        tick_index += 1

    training_summary = {
        "train_raw_budget": int(train_raw_budget),
        "train_patch_budget": int(train_patch_budget),
        "train_focus_budget": int(train_focus_budget),
        "trained_epochs": int(trained_epochs),
        "stages": stages,
        "accepted": bool(stages and stages[-1].get("accepted", False)),
        "stabilize_ticks": int(stabilize_ticks),
    }
    return runtime, training_summary


def _run_condition_for_pair(
    *,
    payload: dict[str, Any],
    pair: OCRPair,
    distractor_texts: list[str],
    image_bytes: bytes,
    condition: ProbeCondition,
    raw_budget: int,
    patch_budget: int,
    focus_budget: int,
    dynamic_summary_limit: int,
    canvas_size: tuple[int, int],
    read_only: bool,
) -> dict[str, Any]:
    runtime = _probe_runtime(
        payload=payload,
        raw_budget=raw_budget,
        patch_budget=patch_budget,
        focus_budget=focus_budget,
        dynamic_enabled=condition.dynamic_enabled,
        auto_reorient_enabled=condition.auto_reorient_enabled,
        dynamic_summary_limit=dynamic_summary_limit,
        read_only=read_only,
    )
    rows: list[dict[str, Any]] = []
    for tick_index, dx in enumerate(condition.positions):
        shifted = _shift_image_bytes(image_bytes, dx=int(dx), size=canvas_size)
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            image_bytes=shifted,
            source_type=f"dynamic_ocr::{condition.name}::{pair.pair_id}",
            execute_selected_actions=condition.execute_selected_actions,
        )
        final_eval = _evaluate_probe(
            tick=tick,
            target_text=pair.text_label,
            distractor_texts=distractor_texts,
        )
        rules_result = dict(tick.get("rules_result", {}) or {})
        image_packet = dict(tick.get("image_packet", {}) or {})
        dynamic_summary = dict(image_packet.get("dynamic_track_summary", {}) or {})
        runtime_effects = dict(tick.get("runtime_action_effects", {}) or {})
        habituation = dict(rules_result.get("cognitive_feeling_habituation", {}) or {})
        gains = dict(habituation.get("gains", {}) or {})
        recall_query_preview = dict(tick.get("recall_query_preview", {}) or {})
        bn_head = [dict(item) for item in (tick.get("bn_list", []) or [])[:4] if isinstance(item, dict)]
        rows.append(
            {
                "tick_index": tick_index,
                "dx": int(dx),
                "elapsed_ms": _round4(elapsed_ms),
                "strict_success": bool(final_eval.get("strict_success", False)),
                "bn_success": bool(final_eval.get("bn_success", False)),
                "cstar_success": bool(final_eval.get("cstar_success", False)),
                "state_success": bool(final_eval.get("state_success", False)),
                "bn_best_text": str(final_eval.get("bn_best_text", "") or ""),
                "bn_target_rank": int(final_eval.get("bn_target_rank", 0) or 0),
                "cstar_best_text": str(final_eval.get("cstar_best_text", "") or ""),
                "cstar_margin": _round4(float(final_eval.get("cstar_margin", 0.0) or 0.0)),
                "state_best_text": str(final_eval.get("state_best_text", "") or ""),
                "state_margin": _round4(float(final_eval.get("state_margin", 0.0) or 0.0)),
                "surprise": _round4(float((rules_result.get("emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)),
                "raw_surprise": _round4(float((rules_result.get("raw_emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)),
                "dissonance": _round4(float((rules_result.get("emotion_channels", {}) or {}).get("dissonance", 0.0) or 0.0)),
                "raw_dissonance": _round4(float((rules_result.get("raw_emotion_channels", {}) or {}).get("dissonance", 0.0) or 0.0)),
                "surprise_gain": _round4(float(gains.get("surprise", 1.0) or 1.0)),
                "dissonance_gain": _round4(float(gains.get("dissonance", 1.0) or 1.0)),
                "auto_reorient": bool(rules_result.get("auto_visual_reorient")),
                "selected_actions": [
                    str(item.get("action_name", "") or "")
                    for item in (rules_result.get("planned_selected_actions_preview", []) or [])
                    if isinstance(item, dict)
                ],
                "raw_count": int(image_packet.get("total_patch_count", 0) or 0),
                "raw_state_budget": int(image_packet.get("raw_state_budget", 0) or 0),
                "memory_write_count": int(len(image_packet.get("memory_write_samples", []) or [])),
                "focus_priority_count": int(len(image_packet.get("focus_priority_samples", []) or [])),
                "dynamic_object_count": int(dynamic_summary.get("object_count", 0) or 0),
                "dynamic_track_count": int(dynamic_summary.get("track_count", 0) or 0),
                "gaze_after": dict(runtime_effects.get("gaze_center_after", {}) or {}),
                "recall_query_source_histogram": dict(recall_query_preview.get("source_histogram", {}) or {}),
                "recall_query_channel_histogram": dict(recall_query_preview.get("channel_histogram", {}) or {}),
                "recall_query_preview": [dict(item) for item in (recall_query_preview.get("preview", []) or [])[:8] if isinstance(item, dict)],
                "bn_head": [
                    {
                        "memory_id": str(item.get("memory_id", "") or ""),
                        "memory_kind": str(item.get("memory_kind", "") or ""),
                        "text": str(item.get("text", "") or ""),
                        "score": _round4(float(item.get("score", 0.0) or 0.0)),
                        "raw_score": _round4(float(item.get("raw_score", 0.0) or 0.0)),
                        "overlap_labels": list(item.get("overlap_labels", []) or [])[:12],
                        "candidate_sources": list(item.get("candidate_sources", []) or [])[:8],
                        "query_vector_tokens": list(item.get("query_vector_tokens", []) or [])[:8],
                        "vector_tokens": list(item.get("vector_tokens", []) or [])[:8],
                        "score_breakdown": dict(item.get("score_breakdown", {}) or {}),
                    }
                    for item in bn_head
                ],
            }
        )

    first_success_tick = next((row["tick_index"] for row in rows if bool(row.get("strict_success", False))), None)
    return {
        "pair_id": pair.pair_id,
        "target_text_label": pair.text_label,
        "condition": condition.name,
        "first_success_tick": first_success_tick,
        "success_count": int(sum(1 for row in rows if bool(row.get("strict_success", False)))),
        "mean_elapsed_ms": _round4(sum(float(row["elapsed_ms"]) for row in rows) / max(1, len(rows))),
        "rows": rows,
    }


def _render_report(*, training: dict[str, Any], probe_rows: list[dict[str, Any]], raw_budget: int) -> str:
    lines: list[str] = []
    lines.append("# V2 动态视觉 OCR 联动实验")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().isoformat()}")
    lines.append(f"- 训练底座: raw={int(training.get('train_raw_budget', 0) or 0)} / memory={int(training.get('train_patch_budget', 0) or 0)} / focus={int(training.get('train_focus_budget', 0) or 0)} / epoch={int(training.get('trained_epochs', 0) or 0)}")
    lines.append(f"- 探测预算: raw={int(raw_budget)} / memory=8 / focus=4")
    lines.append("")
    lines.append("## 条件说明")
    lines.append("- `static_center_passive`: 静止、无动态摘要、无自动惊回看、无动作执行。")
    lines.append("- `moving_passive_no_dynamic`: 目标移动、无动态摘要、无自动惊回看、无动作执行。")
    lines.append("- `moving_dynamic_no_auto`: 目标移动、有动态摘要、无自动惊回看、无动作执行。")
    lines.append("- `moving_dynamic_auto`: 目标移动、有动态摘要、有自动惊回看、无动作执行。")
    lines.append("- `moving_dynamic_full`: 目标移动、有动态摘要、有自动惊回看、且执行内部动作。")
    lines.append("")
    lines.append("## 结果总览")
    for row in probe_rows:
        lines.append(
            f"- `{row['condition']}` / `{row['pair_id']}` -> first_success_tick={row['first_success_tick']} / "
            f"success_count={row['success_count']} / mean_elapsed_ms={row['mean_elapsed_ms']}"
        )
    lines.append("")
    lines.append("## 解释边界")
    lines.append("- 本实验关注的是低预算下，动态视觉联动是否能让 OCR-like 识别更早、更稳地成功。")
    lines.append("- 它不是通用 OCR 基准，也不证明跨字体、跨噪声、跨场景的泛化。")
    lines.append("- 如果 `moving_dynamic_auto` 或 `moving_dynamic_full` 明显早于 `moving_passive_no_dynamic` 成功，则可以证明动态视觉链对识别稳定性有正贡献。")
    return "\n".join(lines)


def _write_probe_checkpoint(
    *,
    output_dir: Path,
    training: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    probe_raw_budget: int,
    probe_patch_budget: int,
    probe_focus_budget: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_epochs": int(training.get("trained_epochs", 0) or 0),
        "train_raw_budget": int(training.get("train_raw_budget", 0) or 0),
        "train_patch_budget": int(training.get("train_patch_budget", 0) or 0),
        "train_focus_budget": int(training.get("train_focus_budget", 0) or 0),
        "training_accepted": bool(training.get("accepted", False)),
        "training_stages": list(training.get("stages", []) or []),
        "probe_raw_budget": int(probe_raw_budget),
        "probe_patch_budget": int(probe_patch_budget),
        "probe_focus_budget": int(probe_focus_budget),
        "probe_rows": probe_rows,
        "meta": dict(meta or {}),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_text(output_dir / "report.md", _render_report(training=training, probe_rows=probe_rows, raw_budget=probe_raw_budget))
    return summary


def run_probe(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _parse_args()
    suffix = f"_{args.output_tag.strip()}" if str(args.output_tag or "").strip() else ""
    output_dir = OUTPUT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _select_pairs(list(args.pairs or []))
    if not pairs:
        raise SystemExit("No pairs selected.")
    positions_override = _parse_int_tuple(str(args.positions or ""), fallback=())
    conditions = _select_conditions(
        list(args.conditions or []),
        positions_override=positions_override if positions_override else None,
        dynamic_summary_limit=int(args.dynamic_summary_limit),
    )
    if not conditions:
        raise SystemExit("No conditions selected.")
    train_plan = _parse_int_tuple(str(args.train_plan or ""), fallback=(8, 4, 4, 2))
    image_map = {pair.pair_id: _render_handwritten_image(pair) for pair in pairs}
    trained_runtime, training = _train_runtime(
        pairs=pairs,
        image_map=image_map,
        train_plan=train_plan,
        train_raw_budget=int(args.train_raw_budget),
        train_patch_budget=int(args.train_patch_budget),
        train_focus_budget=int(args.train_focus_budget),
        stabilize_ticks=int(args.stabilize_ticks),
    )
    payload = dict(trained_runtime.export_payload() or {})
    raw_budget = int(args.probe_raw_budget)
    patch_budget = int(args.probe_patch_budget)
    focus_budget = int(args.probe_focus_budget)
    canvas_size = (int(args.canvas_width), int(args.canvas_height))
    started = time.perf_counter()
    probe_rows: list[dict[str, Any]] = []
    checkpoint_meta = {
        "selected_pairs": [pair.pair_id for pair in pairs],
        "selected_conditions": [condition.name for condition in conditions],
        "train_plan": list(train_plan),
        "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
        "dynamic_summary_limit": int(args.dynamic_summary_limit),
        "probe_read_only": bool(args.probe_read_only),
    }
    _write_json(output_dir / "config_snapshot.json", checkpoint_meta)
    for condition in conditions:
        for pair in pairs:
            row = _run_condition_for_pair(
                payload=payload,
                pair=pair,
                distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
                image_bytes=image_map[pair.pair_id],
                condition=condition,
                raw_budget=raw_budget,
                patch_budget=patch_budget,
                focus_budget=focus_budget,
                dynamic_summary_limit=int(args.dynamic_summary_limit),
                canvas_size=canvas_size,
                read_only=bool(args.probe_read_only),
            )
            probe_rows.append(row)
            _write_json(output_dir / f"partial_{condition.name}_{pair.pair_id}.json", row)
            _write_probe_checkpoint(
                output_dir=output_dir,
                training=training,
                probe_rows=probe_rows,
                probe_raw_budget=raw_budget,
                probe_patch_budget=patch_budget,
                probe_focus_budget=focus_budget,
                meta=checkpoint_meta,
            )
    total_elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = _write_probe_checkpoint(
        output_dir=output_dir,
        training=training,
        probe_rows=probe_rows,
        probe_raw_budget=raw_budget,
        probe_patch_budget=patch_budget,
        probe_focus_budget=focus_budget,
        meta={**checkpoint_meta, "probe_total_elapsed_ms": _round4(total_elapsed_ms)},
    )
    return {"output_dir": str(output_dir), "summary": summary}


def main() -> None:
    result = run_probe(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
