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

from core.runtime_v2 import RuntimeV2  # noqa: E402
from observatory_v2.config import load_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "attention_reversal_probe"


@dataclass(frozen=True)
class ProbeRun:
    probe_id: str
    description: str
    firmness_norm: float
    old_text: str
    new_text: str
    old_ticks: int
    new_ticks: int


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_text_tick(runtime: RuntimeV2, *, tick_index: int, text: str, source_type: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    tick = runtime.process_text_tick(text=text, tick_index=tick_index, source_type=source_type)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.set_last_logic_ms(elapsed_ms)
    return tick, elapsed_ms


def _top_query_labels(tick: dict[str, Any], limit: int = 8) -> list[str]:
    preview = list(((tick.get("recall_query_preview", {}) or {}).get("preview", []) or []))
    return [str(item.get("sa_label", "") or "") for item in preview[:limit]]


def _run_probe(probe: ProbeRun) -> dict[str, Any]:
    runtime = RuntimeV2(config=load_config())
    rows: list[dict[str, Any]] = []
    tick_index = 0

    for _ in range(probe.old_ticks):
        tick, elapsed_ms = _run_text_tick(runtime, tick_index=tick_index, text=probe.old_text, source_type=f"probe::{probe.probe_id}::old")
        rows.append(
            {
                "tick_index": tick_index,
                "phase": "old_context",
                "text": probe.old_text,
                "elapsed_ms": _round4(elapsed_ms),
                "query_top_labels": _top_query_labels(tick),
                "focus_text": str((tick.get("a_focus", {}) or {}).get("focus_text", "") or ""),
            }
        )
        tick_index += 1

    handoff_tick, elapsed_ms = _run_text_tick(runtime, tick_index=tick_index, text=probe.new_text, source_type=f"probe::{probe.probe_id}::handoff")
    effects = runtime.apply_selected_actions(
        [{"action_name": "continue_focus", "params": {}, "firmness_norm": probe.firmness_norm}],
        runtime_tick=handoff_tick,
    )
    rows.append(
        {
            "tick_index": tick_index,
            "phase": "handoff",
            "text": probe.new_text,
            "elapsed_ms": _round4(elapsed_ms),
            "query_top_labels": _top_query_labels(handoff_tick),
            "focus_text": str((handoff_tick.get("a_focus", {}) or {}).get("focus_text", "") or ""),
            "attention_modulation": dict(effects.get("attention_modulation", {}) or {}),
        }
    )
    tick_index += 1

    for obs_index in range(probe.new_ticks):
        tick, elapsed_ms = _run_text_tick(runtime, tick_index=tick_index, text=probe.new_text, source_type=f"probe::{probe.probe_id}::new")
        labels = _top_query_labels(tick)
        rows.append(
            {
                "tick_index": tick_index,
                "phase": "new_context",
                "observation_index": obs_index,
                "text": probe.new_text,
                "elapsed_ms": _round4(elapsed_ms),
                "query_top_labels": labels,
                "focus_text": str((tick.get("a_focus", {}) or {}).get("focus_text", "") or ""),
                "effective_attention_controls": dict(tick.get("effective_attention_controls", {}) or {}),
                "runtime_controls": dict(tick.get("runtime_controls", {}) or {}),
                "attention_modulation_state": dict(tick.get("attention_modulation_state", {}) or {}),
                "new_wins_query": any(probe.new_text in label for label in labels[:3]),
                "old_still_dominates_query": any("three" in label for label in labels[:2]),
            }
        )
        tick_index += 1

    first_new_tick = next(
        (
            int(row.get("observation_index", -1))
            for row in rows
            if row.get("phase") == "new_context" and bool(row.get("new_wins_query", False))
        ),
        None,
    )
    final_row = next((row for row in reversed(rows) if row.get("phase") == "new_context"), {})
    return {
        "probe_id": probe.probe_id,
        "description": probe.description,
        "firmness_norm": _round4(probe.firmness_norm),
        "old_text": probe.old_text,
        "new_text": probe.new_text,
        "first_new_query_tick": first_new_tick,
        "final_new_wins_query": bool(final_row.get("new_wins_query", False)),
        "final_old_still_dominates_query": bool(final_row.get("old_still_dominates_query", False)),
        "rows": rows,
    }


def _write_report(output_dir: Path, runs: list[dict[str, Any]]) -> None:
    lines = [
        "# V2 注意力坚决程度驱动旧上下文反转实验报告",
        "",
        "## 1. 实验目标",
        "",
        "验证旧上下文主导是否可以主要通过注意力行动本身来被压制，而不是通过额外硬规则。",
        "这里关心的不是删掉旧内容，而是当新输入引发注意力反转时，查询包和焦点能否更快翻到新对象。",
        "",
        "## 2. 实验设计",
        "",
        "先连续输入旧文本形成旧上下文，再输入新文本，并执行一次 `continue_focus`。",
        "该行动的 `firmness_norm` 被映射为下一 tick 的状态池滤波与查询包塑形增益。",
        "",
        "## 3. 结果摘要",
        "",
        "| probe | firmness | first new query tick | final new wins query | final old still dominates |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| {run['probe_id']} | {run['firmness_norm']:.2f} | "
            f"{run['first_new_query_tick'] if run['first_new_query_tick'] is not None else '-'} | "
            f"{'是' if run['final_new_wins_query'] else '否'} | "
            f"{'是' if run['final_old_still_dominates_query'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 4. 解释",
            "",
            "如果更高的坚决程度让新输入更早成为查询顶部、并降低旧上下文最终主导的概率，就说明 AP 可以用统一注意力机制完成上下文反转。",
            "这不是把旧内容打成污染，而是让它在强聚焦下退成背景联想。",
        ]
    )
    output_dir.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = [
        ProbeRun("firm_low", "低坚决度", 0.35, "three three three", "eight", 4, 6),
        ProbeRun("firm_mid", "中坚决度", 0.85, "three three three", "eight", 4, 6),
        ProbeRun("firm_high", "高坚决度", 1.25, "three three three", "eight", 4, 6),
    ]
    runs = [_run_probe(probe) for probe in probes]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "probe_count": len(runs),
        "runs": runs,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir, runs)
    print(json.dumps({"output_dir": str(output_dir), "summary_path": str(output_dir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
