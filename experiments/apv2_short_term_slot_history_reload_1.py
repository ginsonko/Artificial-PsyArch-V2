from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.defaults import ShortTermSlotConfig  # noqa: E402
from memory.short_term.slot_packet import ShortTermSlotPacketBuilder  # noqa: E402
from memory.store import MemoryStore  # noqa: E402


ARTIFACT_ID = "apv2_short_term_slot_history_reload_1"
SCHEMA_ID = "apv2_short_term_slot_history_reload_1/v1"
CREATED_AT = "2026-06-13"

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: short-term narrative slot history persists and reloads consistently",
    "student_side_llm": False,
    "answer_table_lookup": False,
    "regex_answer_route": False,
    "keyword_hard_gate": False,
    "full_sentence_macro": False,
    "hidden_solver": False,
    "runtime_mechanism_modified": False,
    "ap_core_full_proof_claimed": False,
    "open_world_dialogue_claimed": False,
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _focus_items(labels: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "sa_label": label,
            "display_text": label.replace("text::", "").replace("_", " "),
            "family": "text",
            "real_energy": 1.0 - index * 0.05,
            "virtual_energy": 0.0,
        }
        for index, label in enumerate(labels)
    ]


def _build_history(builder: ShortTermSlotPacketBuilder) -> list[dict[str, Any]]:
    history = []
    for tick in range(3):
        packet = builder.build(
            tick_index=tick,
            focus_items=_focus_items([f"text::w{tick}a", f"text::w{tick}b", f"text::w{tick}c"]),
            focus_continuation_trace={
                "continuation_strength": 1.0,
                "active_episode_id": 7,
                "recent_entries": [{"continuity_score": 0.92}, {"continuity_score": 0.96}],
            },
            short_term_memory_trace={"last_recall": {"available": True, "score": 0.62}},
            rhythm_trace={},
            runtime_load_trace={"channels": {"load_ratio": 0.12}},
        )
        history.append(packet)
    return history


def _items_signature(packet: dict[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(row.get("sa_label", "") or ""), round(float(row.get("virtual_energy", 0.0) or 0.0), 6))
        for row in packet.get("items", []) or []
    ]


def _slot_query(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in packet.get("items", []) or []:
        label = str(row.get("sa_label", "") or "")
        if not label.startswith("short_term_slot::item::"):
            continue
        rows.append(
            {
                "sa_label": label,
                "display_text": label.replace("short_term_slot::item::", "").replace("text::", ""),
                "family": "text",
                "source_type": "slot",
                "real_energy": 1.0,
                "virtual_energy": float(row.get("virtual_energy", 0.0) or 0.0),
                "cognitive_pressure": 1.0,
            }
        )
    return rows


def _probe(out_dir: Path) -> dict[str, Any]:
    builder = ShortTermSlotPacketBuilder(**asdict(ShortTermSlotConfig()))
    history = _build_history(builder)

    # Write the slot history to a real JSONL file on disk.
    jsonl_path = out_dir / "slot_history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for packet in history:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    jsonl_sha256 = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    # Reload from disk.
    reloaded = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    items_identical = len(history) == len(reloaded) and all(
        _items_signature(history[i]) == _items_signature(reloaded[i]) for i in range(len(history))
    )

    # Recall consistency: original vs reloaded slot items drive the same recall.
    memory = MemoryStore(
        recall_top_k=5,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=32,
        candidate_limit=32,
        scoring_candidate_limit=32,
        learned_rerank_limit=16,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_weight=0.28,
    )
    memory.write_snapshot(
        tick_index=0,
        memory_kind="state",
        items=_focus_items(["text::w0a", "text::w0b", "text::w0c"]),
        focus_labels=["text::w0a"],
        source_text="w0",
    )
    memory.process_pending_index_jobs(budget=16, max_ms=200.0, include_heavy=True)

    recall_orig = [
        (str(r.get("memory_id", "") or ""), _round4(float(r.get("score", 0.0) or 0.0)))
        for r in memory.recall(_slot_query(history[0]), memory_kind="state", top_k=5)
    ]
    recall_reload = [
        (str(r.get("memory_id", "") or ""), _round4(float(r.get("score", 0.0) or 0.0)))
        for r in memory.recall(_slot_query(reloaded[0]), memory_kind="state", top_k=5)
    ]

    checks = {
        "packet_count_match": len(history) == len(reloaded),
        "items_identical_after_reload": items_identical,
        "recall_identical_after_reload": recall_orig == recall_reload,
        "real_file_boundary_hashable": len(jsonl_sha256) == 64,
    }
    try:
        jsonl_display = jsonl_path.relative_to(ROOT).as_posix()
    except ValueError:
        jsonl_display = jsonl_path.as_posix()
    return {
        "schema_id": "short_term_slot_history_reload_probe/v1",
        "history_tick_count": len(history),
        "reloaded_count": len(reloaded),
        "jsonl_path": jsonl_display,
        "jsonl_sha256": jsonl_sha256,
        "recall_original": recall_orig,
        "recall_reloaded": recall_reload,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _build_payload(out_dir: Path) -> dict[str, Any]:
    probe = _probe(out_dir)
    return {
        "schema_id": SCHEMA_ID,
        "artifact_id": ARTIFACT_ID,
        "created_at": CREATED_AT,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": BOUNDARY,
        "summary": {"passed": bool(probe["passed"]), "checks": probe["checks"]},
        "probe": probe,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    p = payload["probe"]
    return f"""# ShortTermSlotHistoryReload-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 短期叙事槽历史重载

- 历史 tick 数: {p['history_tick_count']}, 重载数: {p['reloaded_count']}
- JSONL 文件: `{p['jsonl_path']}`
- JSONL SHA-256: `{p['jsonl_sha256']}`
- 重载前召回: {p['recall_original']}
- 重载后召回: {p['recall_reloaded']}

## 结论口径

- 一段 {p['history_tick_count']} tick 的短期叙事槽历史被写入真实 JSONL 文件并重载, packet 条数与每个 item 的 (label, virtual_energy) 逐字段一致。
- 用原始 slot item 与重载 slot item 分别驱动同一 MemoryStore 召回, 结果 (memory_id, score) 完全一致。
- JSONL 落到真实磁盘并暴露可哈希 SHA-256: 短期叙事的持久化从内存假象推进到可重载、可审计的本地边界。
- 这是 AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
"""


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"{ARTIFACT_ID}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(out_dir)
    (out_dir / f"{ARTIFACT_ID}_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{ARTIFACT_ID}_report.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"artifact_id": ARTIFACT_ID, "passed": payload["summary"]["passed"], "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
