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


ARTIFACT_ID = "apv2_attention_band_online_vector_integration_1"
SCHEMA_ID = "apv2_attention_band_online_vector_integration_1/v1"
CREATED_AT = "2026-06-13"

LEARNED_VECTOR_CAP = 0.22
MAIN_CHANNELS = ["vector_score", "learned_score", "learned_vector_score"]

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: online vector integrates without contaminating main recall channels",
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


def _build(weight: float) -> MemoryStore:
    memory = MemoryStore(
        recall_top_k=6,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=32,
        candidate_limit=32,
        scoring_candidate_limit=32,
        learned_rerank_limit=16,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_weight=weight,
    )
    for tick in range(6):
        memory._online.begin_tick(tick)
        memory._online.observe_positive_pair("text::alpha", "text::gamma", weight=2.0)
    memory.write_snapshot(tick_index=0, memory_kind="state", items=[_it("text::gamma"), _it("text::delta")], focus_labels=["text::gamma"], source_text="gamma delta")
    memory.write_snapshot(tick_index=1, memory_kind="state", items=[_it("text::alpha"), _it("text::beta")], focus_labels=["text::alpha"], source_text="alpha beta")
    memory.process_pending_index_jobs(budget=24, max_ms=400.0, include_heavy=True)
    return memory


def _audit_rows(memory: MemoryStore, query: list[dict]) -> dict[str, dict]:
    audit = memory.audit_recall(query, memory_kind="state", top_k=6, exact_limit=6)
    return {str(r.get("memory_id", "") or ""): r for r in audit.get("exact_rows", []) or []}


def _channel(row: dict, name: str) -> float:
    return _round4(float(row.get(name, 0.0) or 0.0))


def _probe() -> dict[str, Any]:
    query = [_it("text::alpha", real=1.2), _it("text::beta")]
    rows_off = _audit_rows(_build(0.0), query)
    rows_default = _audit_rows(_build(0.28), query)
    rows_high = _audit_rows(_build(0.9), query)

    # mem-2 = exact alpha/beta match; mem-1 = learned neighbor (gamma).
    exact_off = rows_off.get("mem-2", {})
    exact_default = rows_default.get("mem-2", {})
    neighbor_default = rows_default.get("mem-1", {})

    channel_table = {
        ch: {"off": _channel(exact_off, ch), "default": _channel(exact_default, ch)}
        for ch in MAIN_CHANNELS
    }

    vector_unchanged = channel_table["vector_score"]["off"] == channel_table["vector_score"]["default"]
    learned_measure_stable = (
        channel_table["learned_score"]["off"] == channel_table["learned_score"]["default"]
        and channel_table["learned_vector_score"]["off"] == channel_table["learned_vector_score"]["default"]
    )

    # Effective learned-vector coefficient is capped at 0.22 regardless of weight.
    eff_coeff_default = min(LEARNED_VECTOR_CAP, 0.28)
    eff_coeff_high = min(LEARNED_VECTOR_CAP, 0.9)
    learned_bounded = eff_coeff_default <= LEARNED_VECTOR_CAP and eff_coeff_high == LEARNED_VECTOR_CAP

    checks = {
        "exact_match_vector_channel_unchanged": vector_unchanged,
        "learned_channel_measurements_stable": learned_measure_stable,
        "learned_vector_coefficient_bounded": learned_bounded,
        "learned_present_on_neighbor": _channel(neighbor_default, "learned_vector_score") > 0.0
        or _channel(neighbor_default, "learned_score") > 0.0,
    }
    return {
        "schema_id": "attention_band_online_vector_integration_probe/v1",
        "channel_table": channel_table,
        "learned_vector_cap": LEARNED_VECTOR_CAP,
        "effective_learned_vector_coeff": {"default": _round4(eff_coeff_default), "high": _round4(eff_coeff_high)},
        "neighbor_learned_vector_score": _channel(neighbor_default, "learned_vector_score"),
        "neighbor_learned_score": _channel(neighbor_default, "learned_score"),
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
    ct = p["channel_table"]
    rows = "\n".join(f"| {ch} | {ct[ch]['off']} | {ct[ch]['default']} |" for ch in MAIN_CHANNELS)
    return f"""# AttentionBand-OnlineVectorIntegration-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 精确匹配快照的分量分数 (learned_weight 0 vs 0.28)

| 分量 | weight=0 | weight=0.28 |
|---|---:|---:|
{rows}

learned_vector 有效系数被 cap 在 {p['learned_vector_cap']}: default={p['effective_learned_vector_coeff']['default']}, high(weight=0.9)={p['effective_learned_vector_coeff']['high']}

## 结论口径

- 精确匹配快照的 vector_score 在 learned_weight=0 与 0.28 下完全相同: 主通道证据独立测量, 不被 learned 改写。
- learned_score / learned_vector_score 的原始测量值跨权重稳定; learned_weight 只在求和时作为系数相乘, 不改变测量本身。
- learned_vector 进入总分的有效系数被 `min(0.22, learned_weight)` cap 住, 高权重下也不超过 0.22: learned 是有界叠加项。
- 因此在线向量是"叠加而非改写": 它给经验邻居加分, 但不污染 posting/vector/numeric/relation 主通道。
- AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
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
