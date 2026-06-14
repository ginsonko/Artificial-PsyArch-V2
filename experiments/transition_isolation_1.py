from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.embedding.online_store import OnlineEmbeddingStore  # noqa: E402
from memory.store import MemoryStore  # noqa: E402


ARTIFACT_ID = "transition_isolation_1"
SCHEMA_ID = "transition_isolation_1/v1"
CREATED_AT = "2026-06-13"

UNCHANGED_TOLERANCE = 1e-6

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: transition strengthens successor without contaminating concept vectors",
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


def _item(label: str, *, real: float = 1.0, virtual: float = 0.0) -> dict[str, Any]:
    return {
        "sa_label": label,
        "display_text": label.replace("text::", "").replace("_", " "),
        "family": "text",
        "source_type": "external_text",
        "real_energy": float(real),
        "virtual_energy": float(virtual),
        "cognitive_pressure": float(real) - float(virtual),
    }


def _direct_isolation_probe() -> dict[str, Any]:
    store = OnlineEmbeddingStore(dim=32, token_limit=128, min_support_to_promote=1, per_tick_update_limit=64)

    # A and B each gain independent existence (promotable), with no link between them.
    for tick in range(2):
        store.begin_tick(tick)
        store.observe_positive_pair("text::A", "text::A_ctx", weight=1.0)
        store.observe_positive_pair("text::B", "text::B_ctx", weight=1.0)

    vec_sim_before = store.learned_vector_similarity(["text::A"], ["text::B"])["score"]
    sim_before = store.learned_similarity(["text::A"], ["text::B"])["score"]
    transition_before = store.learned_transition(["text::A"], ["text::B"])["score"]

    # Train A -> B successor only (directed transition, no co-occurrence).
    for tick in range(2, 18):
        store.begin_tick(tick)
        store.observe_transition_pair("text::A", "text::B", weight=2.0)

    vec_sim_after = store.learned_vector_similarity(["text::A"], ["text::B"])["score"]
    sim_after = store.learned_similarity(["text::A"], ["text::B"])["score"]
    transition_after = store.learned_transition(["text::A"], ["text::B"])["score"]
    transition_reverse = store.learned_transition(["text::B"], ["text::A"])["score"]
    evidence = store.pair_evidence("text::A", "text::B")

    checks = {
        "transition_rises": transition_after > transition_before,
        "concept_vector_not_contaminated": abs(vec_sim_after - vec_sim_before) < UNCHANGED_TOLERANCE,
        "cooccurrence_similarity_not_contaminated": abs(sim_after - sim_before) < UNCHANGED_TOLERANCE,
        "transition_is_directed": transition_after > transition_reverse,
        "evidence_is_pure_transition": float(evidence.get("transition_raw", 0.0) or 0.0) > 0.0
        and float(evidence.get("positive_raw", 0.0) or 0.0) == 0.0,
    }
    return {
        "schema_id": "transition_isolation_direct_probe/v1",
        "scores": {
            "vec_sim_before": _round4(vec_sim_before),
            "vec_sim_after": _round4(vec_sim_after),
            "sim_before": _round4(sim_before),
            "sim_after": _round4(sim_after),
            "transition_before": _round4(transition_before),
            "transition_after": _round4(transition_after),
            "transition_reverse": _round4(transition_reverse),
        },
        "pair_evidence": {
            "transition_raw": _round4(float(evidence.get("transition_raw", 0.0) or 0.0)),
            "positive_raw": _round4(float(evidence.get("positive_raw", 0.0) or 0.0)),
            "vector_similarity": _round4(float(evidence.get("vector_similarity", 0.0) or 0.0)),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _recall_successor_probe() -> dict[str, Any]:
    memory = MemoryStore(
        recall_top_k=8,
        predict_top_k=5,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=32,
        candidate_limit=32,
        scoring_candidate_limit=32,
        learned_rerank_limit=16,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_weight=0.28,
        transition_learned_weight=0.18,
    )
    # Repeated consecutive A -> B snapshots build a snapshot-level transition edge.
    for rep in range(4):
        memory.write_snapshot(tick_index=rep * 2, memory_kind="state", items=[_item("text::A")], focus_labels=["text::A"], source_text="A")
        memory.write_snapshot(tick_index=rep * 2 + 1, memory_kind="state", items=[_item("text::B")], focus_labels=["text::B"], source_text="B")
    memory.process_pending_index_jobs(budget=32, max_ms=400.0, include_heavy=True)

    a_id = None
    for memory_id, snapshot in memory._snapshot_by_id.items():
        if any(it.get("sa_label") == "text::A" for it in snapshot.get("items", []) or []):
            a_id = memory_id
            break

    rows = memory.successors(str(a_id or ""), memory_kind="state", top_k=5) if a_id else []
    top = rows[0] if rows else {}
    learned_transition_score = float(top.get("learned_transition_score", 0.0) or 0.0)
    checks = {
        "successor_recall_exposes_positive_transition": learned_transition_score > 0.0,
    }
    return {
        "schema_id": "transition_isolation_recall_probe/v1",
        "source_a_id": str(a_id or ""),
        "top_successor_id": str(top.get("successor_memory_id", "") or ""),
        "top_successor_score": _round4(float(top.get("score", 0.0) or 0.0)),
        "top_learned_transition_score": _round4(learned_transition_score),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _build_payload() -> dict[str, Any]:
    direct = _direct_isolation_probe()
    recall = _recall_successor_probe()
    checks = {
        "direct_isolation_probe_passed": bool(direct["passed"]),
        "recall_successor_probe_passed": bool(recall["passed"]),
    }
    return {
        "schema_id": SCHEMA_ID,
        "artifact_id": ARTIFACT_ID,
        "created_at": CREATED_AT,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": BOUNDARY,
        "summary": {"passed": all(checks.values()), "checks": checks},
        "direct_isolation_probe": direct,
        "recall_successor_probe": recall,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    d = payload["direct_isolation_probe"]
    r = payload["recall_successor_probe"]
    s = d["scores"]
    return f"""# TransitionIsolation-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 直接隔离 (A -> B transition 训练前后)

| 量 | 训练前 | 训练后 |
|---|---:|---:|
| 后继 learned_transition(A->B) | {s['transition_before']} | {s['transition_after']} |
| 概念向量 learned_vector_similarity(A,B) | {s['vec_sim_before']} | {s['vec_sim_after']} |
| 共现 learned_similarity(A,B) | {s['sim_before']} | {s['sim_after']} |
| 反向后继 learned_transition(B->A) | - | {s['transition_reverse']} |

pair_evidence(A,B): transition_raw={d['pair_evidence']['transition_raw']}, positive_raw={d['pair_evidence']['positive_raw']}

## 召回侧后继 (successors API)

source A = {r['source_a_id']}, top successor = {r['top_successor_id']}, score = {r['top_successor_score']}, learned_transition_score = {r['top_learned_transition_score']}

## 结论口径

- 大量训练 A->B 后继, learned_transition(A->B) 从 {s['transition_before']} 升到 {s['transition_after']}。
- 同时 A、B 的对称概念相似度完全不变 (vector {s['vec_sim_before']} -> {s['vec_sim_after']}, similarity {s['sim_before']} -> {s['sim_after']}): transition 增强后继, 不污染概念向量。
- 后继是有向的: A->B ({s['transition_after']}) 远大于 B->A ({s['transition_reverse']})。
- pair_evidence 为纯 transition (transition_raw>0, positive_raw=0)。
- 召回侧 successors API 暴露正的 learned_transition_score, 后继增益可审计。
- 这是 AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
"""


def main() -> None:
    payload = _build_payload()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"{ARTIFACT_ID}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ARTIFACT_ID}_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{ARTIFACT_ID}_report.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"artifact_id": ARTIFACT_ID, "passed": payload["summary"]["passed"], "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
