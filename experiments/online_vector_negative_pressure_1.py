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


ARTIFACT_ID = "online_vector_negative_pressure_1"
SCHEMA_ID = "online_vector_negative_pressure_1/v1"
CREATED_AT = "2026-06-13"

GOOD_PRESERVE_TOLERANCE = 0.05

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: negative pressure prunes wrong residue",
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


def _item(
    label: str,
    *,
    real: float = 1.0,
    virtual: float = 0.0,
    family: str = "text",
    source_type: str = "external_text",
) -> dict[str, Any]:
    return {
        "sa_label": label,
        "display_text": label.replace("text::", "").replace("_", " "),
        "family": family,
        "source_type": source_type,
        "real_energy": float(real),
        "virtual_energy": float(virtual),
        "cognitive_pressure": float(real) - float(virtual),
    }


def _direct_pressure_probe() -> dict[str, Any]:
    store = OnlineEmbeddingStore(dim=32, token_limit=128, min_support_to_promote=1, per_tick_update_limit=64)

    # Build the correct association: C_real -> S_good.
    for tick in range(4):
        store.begin_tick(tick)
        store.observe_positive_pair("text::C_real", "text::S_good", weight=2.0)
    good_before = store.learned_vector_similarity(["text::C_real"], ["text::S_good"])["score"]

    # Early noise: a wrong subject also picked up a weak positive link (stale residue).
    for tick in range(4, 6):
        store.begin_tick(tick)
        store.observe_positive_pair("text::C_real", "text::S_wrong", weight=1.0)
    wrong_before = store.learned_vector_similarity(["text::S_wrong"], ["text::C_real"])["score"]

    # Repeated over-prediction of the wrong subject under C_real, each a mismatch
    # producing negative cognitive pressure (directed: only the wrong subject moves).
    for tick in range(6, 16):
        store.begin_tick(tick)
        store.observe_negative_anchor("text::S_wrong", "text::C_real", weight=2.0)

    wrong_after = store.learned_vector_similarity(["text::S_wrong"], ["text::C_real"])["score"]
    good_after = store.learned_vector_similarity(["text::C_real"], ["text::S_good"])["score"]
    evidence = store.pair_evidence("text::S_wrong", "text::C_real")

    checks = {
        "wrong_residue_pushed_down": wrong_after < wrong_before,
        "wrong_residue_now_non_positive": wrong_after <= 0.0,
        "good_association_preserved": good_after >= good_before - GOOD_PRESERVE_TOLERANCE,
        "negative_evidence_visible": float(evidence.get("negative_raw", 0.0) or 0.0) > 0.0
        and float(evidence.get("source_negative_support", 0.0) or 0.0) > 0.0,
    }
    return {
        "schema_id": "negative_pressure_direct_probe/v1",
        "scores": {
            "good_before": _round4(good_before),
            "good_after": _round4(good_after),
            "wrong_before": _round4(wrong_before),
            "wrong_after": _round4(wrong_after),
            "good_drop": _round4(good_before - good_after),
            "wrong_drop": _round4(wrong_before - wrong_after),
        },
        "pair_evidence": {
            "negative_raw": _round4(float(evidence.get("negative_raw", 0.0) or 0.0)),
            "source_negative_support": _round4(float(evidence.get("source_negative_support", 0.0) or 0.0)),
            "vector_similarity": _round4(float(evidence.get("vector_similarity", 0.0) or 0.0)),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _recall_competition_probe() -> dict[str, Any]:
    memory = MemoryStore(
        recall_top_k=8,
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

    # Train correct association and stale wrong residue, then apply negative pressure.
    for tick in range(4):
        memory._online.begin_tick(tick)
        memory._online.observe_positive_pair("text::C_real", "text::S_good", weight=2.0)
    for tick in range(4, 6):
        memory._online.begin_tick(tick)
        memory._online.observe_positive_pair("text::C_real", "text::S_wrong", weight=1.0)
    for tick in range(6, 16):
        memory._online.begin_tick(tick)
        memory._online.observe_negative_anchor("text::S_wrong", "text::C_real", weight=2.0)

    # good_snapshot subject = S_good; wrong_snapshot subject = S_wrong.
    memory.write_snapshot(
        tick_index=0,
        memory_kind="state",
        items=[_item("text::S_good"), _item("text::tail_good")],
        focus_labels=["text::S_good"],
        source_text="s good tail",
    )
    memory.write_snapshot(
        tick_index=1,
        memory_kind="state",
        items=[_item("text::S_wrong"), _item("text::tail_wrong")],
        focus_labels=["text::S_wrong"],
        source_text="s wrong tail",
    )
    memory.process_pending_index_jobs(budget=24, max_ms=400.0, include_heavy=True)

    query = [_item("text::C_real", real=1.2)]
    audit = memory.audit_recall(query, memory_kind="state", top_k=8, exact_limit=8)
    rows = list(audit.get("exact_rows", []) or [])

    def lvs(memory_id: str) -> float:
        for row in rows:
            if str(row.get("memory_id", "") or "") == memory_id:
                return float(row.get("learned_vector_score", 0.0) or 0.0)
        return 0.0

    good_lvs = lvs("mem-1")
    wrong_lvs = lvs("mem-2")
    checks = {
        "correct_subject_outscores_wrong_residue": good_lvs > wrong_lvs,
    }
    return {
        "schema_id": "negative_pressure_recall_probe/v1",
        "good_snapshot_learned_vector_score": _round4(good_lvs),
        "wrong_snapshot_learned_vector_score": _round4(wrong_lvs),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _build_payload() -> dict[str, Any]:
    direct = _direct_pressure_probe()
    recall = _recall_competition_probe()
    checks = {
        "direct_pressure_probe_passed": bool(direct["passed"]),
        "recall_competition_probe_passed": bool(recall["passed"]),
    }
    return {
        "schema_id": SCHEMA_ID,
        "artifact_id": ARTIFACT_ID,
        "created_at": CREATED_AT,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": BOUNDARY,
        "summary": {"passed": all(checks.values()), "checks": checks},
        "direct_pressure_probe": direct,
        "recall_competition_probe": recall,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    d = payload["direct_pressure_probe"]
    r = payload["recall_competition_probe"]
    s = d["scores"]
    return f"""# OnlineVector-NegativePressure-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 直接负压学习 (learned vector 相似度)

| 量 | 值 |
|---|---:|
| 正确关联 C_real-S_good 负压前 | {s['good_before']} |
| 正确关联 C_real-S_good 负压后 | {s['good_after']} (下降 {s['good_drop']}) |
| 错误残留 S_wrong-C_real 负压前 | {s['wrong_before']} |
| 错误残留 S_wrong-C_real 负压后 | {s['wrong_after']} (下降 {s['wrong_drop']}) |

pair_evidence(S_wrong, C_real): negative_raw={d['pair_evidence']['negative_raw']}, source_negative_support={d['pair_evidence']['source_negative_support']}

## 召回竞争 (audit 路径 learned_vector_score)

| 快照 | learned_vector_score |
|---|---:|
| 正确 subject 快照 (S_good) | {r['good_snapshot_learned_vector_score']} |
| 错误残留快照 (S_wrong) | {r['wrong_snapshot_learned_vector_score']} |

## 结论口径

- 反复过预测错误 subject 产生的负认知压, 把错误残留 S_wrong 从正相似 ({s['wrong_before']}) 推到非正 ({s['wrong_after']})。
- 同时正确关联 C_real-S_good 基本保持 ({s['good_before']} -> {s['good_after']}), 负压靶向错误残留而不误伤有用经验。
- 在 audit 召回竞争中, 以 C_real 为 query, 正确 subject 快照的 learned-vector 贡献高于错误残留快照。
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
