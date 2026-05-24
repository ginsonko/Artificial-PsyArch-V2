# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v2_vision_ocr_probe import (  # noqa: E402
    OCRPair,
    _base_overrides,
    _budget_triplet,
    _build_runtime,
    _evaluate_probe,
    _inject_reward,
    _render_handwritten_image,
    _round4,
    _run_multimodal_tick,
    _write_json,
    _write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "vision_recall_protocols"


@dataclass(frozen=True)
class RecallProtocol:
    protocol_id: str
    description: str
    reset_between_pairs: bool
    idle_ticks_between_pairs: int
    observation_ticks_per_pair: int
    post_probe_idle_ticks: int


def _protocol_overrides(*, raw_budget: int, patch_budget: int, focus_budget: int) -> dict[str, Any]:
    return {
        **_base_overrides(),
        "vision_raw_state_budget": int(min(256, raw_budget)),
        "vision_patch_budget": int(patch_budget),
        "vision_focus_patch_budget": int(focus_budget),
        "vision_reconstruction_patch_budget": int(max(1024, min(2048, raw_budget * 4))),
    }


def _train_runtime(
    *,
    pairs: list[OCRPair],
    image_map: dict[str, bytes],
    raw_budget: int,
    epochs: int,
    stabilize_ticks: int,
    reward_value: float,
) -> dict[str, Any]:
    raw_budget, patch_budget, focus_budget = _budget_triplet(raw_budget)
    runtime = _build_runtime(overrides=_protocol_overrides(raw_budget=raw_budget, patch_budget=patch_budget, focus_budget=focus_budget))
    tick_index = 0
    train_rows: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        for pair in pairs:
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text=pair.text_label,
                image_bytes=image_map[pair.pair_id],
                source_type=f"vision_protocol_train::{pair.pair_id}",
            )
            _inject_reward(runtime, tick_index=tick_index, tick=tick, pair=pair, reward=reward_value)
            train_rows.append(
                {
                    "tick_index": tick_index,
                    "epoch": epoch,
                    "pair_id": pair.pair_id,
                    "elapsed_ms": _round4(elapsed_ms),
                    "memory_count": int(tick.get("memory_count", 0) or 0),
                    "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
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
            source_type="vision_protocol_stabilize",
        )
        stabilize_rows.append(
            {
                "tick_index": tick_index,
                "elapsed_ms": _round4(elapsed_ms),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
            }
        )
        tick_index += 1
    return {
        "payload": runtime.export_payload(),
        "train_rows": train_rows,
        "stabilize_rows": stabilize_rows,
        "raw_budget": raw_budget,
        "patch_budget": patch_budget,
        "focus_budget": focus_budget,
    }


def _run_idle_ticks(runtime: Any, *, start_tick: int, count: int, source_type: str) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    tick_index = int(start_tick)
    for _ in range(int(count)):
        tick, elapsed_ms = _run_multimodal_tick(
            runtime,
            tick_index=tick_index,
            text="",
            image_bytes=None,
            source_type=source_type,
        )
        rows.append(
            {
                "tick_index": tick_index,
                "elapsed_ms": _round4(elapsed_ms),
                "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
            }
        )
        tick_index += 1
    return tick_index, rows


def _run_protocol(
    *,
    protocol: RecallProtocol,
    training_payload: dict[str, Any],
    pairs: list[OCRPair],
    image_map: dict[str, bytes],
    raw_budget: int,
) -> dict[str, Any]:
    raw_budget, patch_budget, focus_budget = _budget_triplet(raw_budget)
    runtime = _build_runtime(overrides=_protocol_overrides(raw_budget=raw_budget, patch_budget=patch_budget, focus_budget=focus_budget))
    runtime.import_payload({"memory_store": dict(training_payload.get("memory_store", {}) or {})})
    tick_index = 0
    pair_runs: list[dict[str, Any]] = []
    idle_segments: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(pairs):
        if pair_idx > 0:
            if protocol.reset_between_pairs:
                runtime.reset_transient_state(keep_runtime_controls=True)
                runtime.import_payload({"memory_store": dict(training_payload.get("memory_store", {}) or {})})
                idle_segments.append({"kind": "reset_transient_state", "before_pair": pair.pair_id})
            if protocol.idle_ticks_between_pairs > 0:
                tick_index, idle_rows = _run_idle_ticks(
                    runtime,
                    start_tick=tick_index,
                    count=protocol.idle_ticks_between_pairs,
                    source_type=f"vision_protocol_idle_between::{protocol.protocol_id}",
                )
                idle_segments.append({"kind": "idle_between_pairs", "before_pair": pair.pair_id, "rows": idle_rows})

        probe_rows: list[dict[str, Any]] = []
        final_tick: dict[str, Any] = {}
        for _ in range(int(protocol.observation_ticks_per_pair)):
            tick, elapsed_ms = _run_multimodal_tick(
                runtime,
                tick_index=tick_index,
                text="",
                image_bytes=image_map[pair.pair_id],
                source_type=f"vision_protocol_probe::{protocol.protocol_id}::{pair.pair_id}",
            )
            eval_row = _evaluate_probe(
                tick=tick,
                target_text=pair.text_label,
                distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
            )
            probe_rows.append(
                {
                    "tick_index": tick_index,
                    "elapsed_ms": _round4(elapsed_ms),
                    "bn_best_text": eval_row["bn_best_text"],
                    "bn_target_rank": int(eval_row["bn_target_rank"]),
                    "cstar_best_text": eval_row["cstar_best_text"],
                    "cstar_margin": _round4(eval_row["cstar_margin"]),
                    "state_best_text": str(eval_row.get("state_best_text", "") or ""),
                    "state_margin": _round4(float(eval_row.get("state_margin", 0.0) or 0.0)),
                    "strict_success": bool(eval_row["strict_success"]),
                    "focus_has_target": bool(eval_row["focus_has_target"]),
                    "state_pool_size": int((tick.get("state_pool_summary", {}) or {}).get("state_pool_size", 0) or 0),
                }
            )
            final_tick = tick
            tick_index += 1

        final_eval = _evaluate_probe(
            tick=final_tick,
            target_text=pair.text_label,
            distractor_texts=[item.text_label for item in pairs if item.pair_id != pair.pair_id],
        )
        post_idle_rows: list[dict[str, Any]] = []
        if protocol.post_probe_idle_ticks > 0:
            tick_index, post_idle_rows = _run_idle_ticks(
                runtime,
                start_tick=tick_index,
                count=protocol.post_probe_idle_ticks,
                source_type=f"vision_protocol_post_idle::{protocol.protocol_id}::{pair.pair_id}",
            )
        pair_runs.append(
            {
                "pair_id": pair.pair_id,
                "target_text_label": pair.text_label,
                "probe_rows": probe_rows,
                "final_evaluation": final_eval,
                "post_idle_rows": post_idle_rows,
            }
        )

    strict_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("strict_success", False)))
    bn_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("bn_success", False)))
    cstar_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("cstar_success", False)))
    state_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("state_success", False)))
    focus_hits = sum(1 for row in pair_runs if bool((row.get("final_evaluation", {}) or {}).get("focus_success", False)))
    return {
        "protocol_id": protocol.protocol_id,
        "description": protocol.description,
        "raw_budget": raw_budget,
        "observation_ticks_per_pair": int(protocol.observation_ticks_per_pair),
        "idle_ticks_between_pairs": int(protocol.idle_ticks_between_pairs),
        "post_probe_idle_ticks": int(protocol.post_probe_idle_ticks),
        "reset_between_pairs": bool(protocol.reset_between_pairs),
        "strict_accuracy": _round4(strict_hits / max(1, len(pair_runs))),
        "bn_accuracy": _round4(bn_hits / max(1, len(pair_runs))),
        "cstar_accuracy": _round4(cstar_hits / max(1, len(pair_runs))),
        "state_accuracy": _round4(state_hits / max(1, len(pair_runs))),
        "focus_accuracy": _round4(focus_hits / max(1, len(pair_runs))),
        "pair_runs": pair_runs,
        "idle_segments": idle_segments,
    }


def main() -> None:
    output_root = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = [
        OCRPair(pair_id="digit_3", glyph="3", text_label="three", rotate_deg=-6.0),
        OCRPair(pair_id="digit_8", glyph="8", text_label="eight", rotate_deg=5.0),
    ]
    image_map = {pair.pair_id: _render_handwritten_image(pair) for pair in pairs}
    training = _train_runtime(
        pairs=pairs,
        image_map=image_map,
        raw_budget=256,
        epochs=8,
        stabilize_ticks=6,
        reward_value=1.0,
    )

    protocols = [
        RecallProtocol(
            protocol_id="continuous_back_to_back",
            description="先测一个，再直接连续测另一个，中间不清池，只保留每个 probe 后短暂空 tick。",
            reset_between_pairs=False,
            idle_ticks_between_pairs=0,
            observation_ticks_per_pair=4,
            post_probe_idle_ticks=3,
        ),
        RecallProtocol(
            protocol_id="idle_gap_between_pairs",
            description="两次召回之间插入较多空 tick，让状态池自然稳定后再测另一个。",
            reset_between_pairs=False,
            idle_ticks_between_pairs=8,
            observation_ticks_per_pair=4,
            post_probe_idle_ticks=4,
        ),
        RecallProtocol(
            protocol_id="reset_between_pairs",
            description="两次召回之间清空瞬时状态，只保留长期记忆，再测另一个。",
            reset_between_pairs=True,
            idle_ticks_between_pairs=0,
            observation_ticks_per_pair=4,
            post_probe_idle_ticks=3,
        ),
    ]

    protocol_rows = [
        _run_protocol(
            protocol=protocol,
            training_payload=dict(training.get("payload", {}) or {}),
            pairs=pairs,
            image_map=image_map,
            raw_budget=256,
        )
        for protocol in protocols
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_summary": {
            "raw_budget": int(training["raw_budget"]),
            "patch_budget": int(training["patch_budget"]),
            "focus_budget": int(training["focus_budget"]),
            "train_tick_count": len(training["train_rows"]),
            "stabilize_tick_count": len(training["stabilize_rows"]),
            "mean_train_elapsed_ms": _round4(
                sum(float(row.get("elapsed_ms", 0.0) or 0.0) for row in training["train_rows"]) / max(1, len(training["train_rows"]))
            ),
        },
        "protocol_rows": protocol_rows,
    }
    report_lines = [
        "# V2 视觉召回协议实验",
        "",
        f"- 训练 raw_budget: {training['raw_budget']}",
        f"- 训练 tick 数: {len(training['train_rows'])}",
        f"- 稳定空 tick 数: {len(training['stabilize_rows'])}",
        "",
        "## 协议结果",
    ]
    for row in protocol_rows:
        report_lines.append(
            f"- `{row['protocol_id']}`: strict_accuracy={row['strict_accuracy']} / "
            f"bn_accuracy={row['bn_accuracy']} / cstar_accuracy={row['cstar_accuracy']} / "
            f"state_accuracy={row['state_accuracy']} / focus_accuracy={row['focus_accuracy']}"
        )
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "training_rows.json", training["train_rows"])
    _write_json(output_root / "stabilize_rows.json", training["stabilize_rows"])
    _write_text(output_root / "report.md", "\n".join(report_lines) + "\n")
    print(json.dumps({"output_dir": str(output_root), "protocol_rows": protocol_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    elapsed = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"total_elapsed_ms": _round4(elapsed)}, ensure_ascii=False))
