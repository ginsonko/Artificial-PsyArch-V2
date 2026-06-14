from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore  # noqa: E402


ARTIFACT_ID = "apv2_residual_b_recall_ordered_completion_1"
SCHEMA_ID = "apv2_residual_b_recall_ordered_completion_1/v1"
CREATED_AT = "2026-06-13"

PARTS = [
    ("AB", ["text::A", "text::B"]),
    ("CD", ["text::C", "text::D"]),
    ("EF", ["text::E", "text::F"]),
    ("GH", ["text::G", "text::H"]),
    ("IJ", ["text::I", "text::J"]),
    ("KL", ["text::K", "text::L"]),
]
QUERY_LABELS = [label for _, labels in PARTS for label in labels]

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: residual B recall absorbs a multi-component query round by round",
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


def _it(label: str, real: float = 1.0) -> dict[str, Any]:
    return {
        "sa_label": label,
        "display_text": label.replace("text::", "").replace("_", " "),
        "family": "text",
        "source_type": "external_text",
        "real_energy": float(real),
        "virtual_energy": 0.0,
        "cognitive_pressure": float(real),
    }


def _probe() -> dict[str, Any]:
    memory = MemoryStore(
        recall_top_k=8,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=64,
        candidate_limit=64,
        scoring_candidate_limit=64,
        learned_rerank_limit=16,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_weight=0.28,
    )
    for index, (name, labels) in enumerate(PARTS):
        memory.write_snapshot(
            tick_index=index,
            memory_kind="state",
            items=[_it(label) for label in labels],
            focus_labels=labels[:1],
            source_text=name,
        )
    memory.process_pending_index_jobs(budget=64, max_ms=600.0, include_heavy=True)

    query = [_it(label) for label in QUERY_LABELS]
    rows = memory.recall_residual(query, memory_kind="state", top_k=6)
    trace = rows[0]["residual_recall_trace"] if rows else []

    rounds = []
    for row in trace:
        rounds.append(
            {
                "round_index": int(row.get("round_index", 0) or 0),
                "winner": str(row.get("winner_memory_id", row.get("memory_id", "")) or ""),
                "matched_labels": list(row.get("matched_labels", []) or []),
                "mass_before": _round4(float(row.get("residual_mass_before", 0.0) or 0.0)),
                "mass_after": _round4(float(row.get("residual_mass_after", 0.0) or 0.0)),
            }
        )

    winners = [r["winner"] for r in rounds]
    befores = [r["mass_before"] for r in rounds]
    afters = [r["mass_after"] for r in rounds]

    checks = {
        "at_least_four_rounds": len(rounds) >= 4,
        "winners_unique": len(winners) == len(set(winners)) and all(winners),
        "each_round_absorbs": all(r["mass_after"] < r["mass_before"] for r in rounds),
        "mass_monotonic_decreasing": all(befores[i + 1] <= befores[i] + 1e-9 for i in range(len(befores) - 1)),
        "matched_labels_valid": all(r["matched_labels"] and set(r["matched_labels"]) <= set(QUERY_LABELS) for r in rounds),
    }
    return {
        "schema_id": "residual_b_recall_ordered_completion_probe/v1",
        "query_component_count": len(QUERY_LABELS),
        "rounds": rounds,
        "mass_curve": {"before": befores, "after": afters},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _build_payload() -> dict[str, Any]:
    probe = _probe()
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
    round_rows = "\n".join(
        f"| {r['round_index']} | {r['winner']} | {', '.join(r['matched_labels'])} | {r['mass_before']} | {r['mass_after']} |"
        for r in p["rounds"]
    )
    return f"""# ResidualBRecall-OrderedCompletion-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 残差吸收轮次 (12 分量 query)

| round | winner | matched labels | mass before | mass after |
|---:|---|---|---:|---:|
{round_rows}

## 结论口径

- 一个 {p['query_component_count']} 分量混合 query 被 {len(p['rounds'])} 轮残差召回逐步吸收: 每轮恰好一个 winner, winner 互不重复。
- residual mass 从 {p['mass_curve']['before'][0]} 单调下降到 {p['mass_curve']['after'][-1]}, 每轮 after < before。
- 每轮 matched_labels 是真实存在于 query 的 SA: 召回有清晰的"轮次感", 而不是把候选压成黑箱相似度列表。
- 这是 AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
"""


def main() -> None:
    payload = _build_payload()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"{ARTIFACT_ID}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ARTIFACT_ID}_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{ARTIFACT_ID}_report.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"artifact_id": ARTIFACT_ID, "passed": payload["summary"]["passed"], "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
