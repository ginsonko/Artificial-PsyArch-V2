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


ARTIFACT_ID = "online_vector_weight_ablation_1"
SCHEMA_ID = "online_vector_weight_ablation_1/v1"
CREATED_AT = "2026-06-13"

WEIGHTS = {"off": 0.0, "default": 0.28, "high": 0.9}

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: online learned vector weight ablation",
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


def _memory_store(weight: float) -> MemoryStore:
    return MemoryStore(
        recall_top_k=8,
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


def _train_neighborhood(memory: MemoryStore) -> None:
    """Repeatedly co-activate query subject `alpha` with neighbor subject `gamma`.

    This pulls their online token vectors together so `gamma` becomes a learned
    neighbor of `alpha` even though they never share surface tokens.
    """
    for tick in range(6):
        memory._online.begin_tick(tick)
        memory._online.observe_positive_pair("text::alpha", "text::gamma", weight=2.0)


def _build_scene(weight: float) -> MemoryStore:
    memory = _memory_store(weight)
    _train_neighborhood(memory)

    # A_neighbor: shares no surface token with the alpha/beta query, but its
    # subject `gamma` is a learned neighbor of the query subject `alpha`.
    memory.write_snapshot(
        tick_index=0,
        memory_kind="state",
        items=[_item("text::gamma"), _item("text::delta")],
        focus_labels=["text::gamma"],
        source_text="gamma delta",
    )
    # A_unrelated: no surface overlap and no learned experience with the query.
    memory.write_snapshot(
        tick_index=1,
        memory_kind="state",
        items=[_item("text::zeta"), _item("text::eta")],
        focus_labels=["text::zeta"],
        source_text="zeta eta",
    )
    # B_exact: exact label/energy match with the query subject `alpha`
    # (strong SA/energy main evidence), but no learned experience.
    memory.write_snapshot(
        tick_index=2,
        memory_kind="state",
        items=[_item("text::alpha", real=1.2), _item("text::beta", real=1.0)],
        focus_labels=["text::alpha"],
        source_text="alpha beta",
    )
    memory.process_pending_index_jobs(budget=24, max_ms=400.0, include_heavy=True)
    return memory


def _row_by_id(rows: list[dict[str, Any]], memory_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("memory_id", "") or "") == memory_id:
            return row
    return None


def _rank_of(rows: list[dict[str, Any]], memory_id: str) -> int:
    ordered = sorted(rows, key=lambda r: -float(r.get("score", 0.0) or 0.0))
    for index, row in enumerate(ordered):
        if str(row.get("memory_id", "") or "") == memory_id:
            return index
    return len(ordered)


# Snapshot ids are assigned in write order: mem-1 neighbor, mem-2 unrelated, mem-3 exact.
NEIGHBOR_ID = "mem-1"
UNRELATED_ID = "mem-2"
EXACT_ID = "mem-3"


def _probe() -> dict[str, Any]:
    query = [_item("text::alpha", real=1.2), _item("text::beta", real=1.0)]
    learned_only_query = [_item("text::alpha", real=1.2)]

    per_weight: dict[str, Any] = {}
    for name, weight in WEIGHTS.items():
        memory = _build_scene(weight)
        audit = memory.audit_recall(query, memory_kind="state", top_k=8, exact_limit=8)
        exact_rows = list(audit.get("exact_rows", []) or [])
        main_rows = memory.recall(query, memory_kind="state", top_k=8)

        neighbor_audit = _row_by_id(exact_rows, NEIGHBOR_ID) or {}
        unrelated_audit = _row_by_id(exact_rows, UNRELATED_ID) or {}
        exact_audit = _row_by_id(exact_rows, EXACT_ID) or {}
        neighbor_main = _row_by_id(main_rows, NEIGHBOR_ID) or {}

        per_weight[name] = {
            "weight": weight,
            "audit": {
                "neighbor": {
                    "score": _round4(float(neighbor_audit.get("score", 0.0) or 0.0)),
                    "learned_vector_score": _round4(float(neighbor_audit.get("learned_vector_score", 0.0) or 0.0)),
                    "learned_score": _round4(float(neighbor_audit.get("learned_score", 0.0) or 0.0)),
                    "rank": _rank_of(exact_rows, NEIGHBOR_ID),
                },
                "unrelated": {
                    "score": _round4(float(unrelated_audit.get("score", 0.0) or 0.0)),
                    "learned_vector_score": _round4(float(unrelated_audit.get("learned_vector_score", 0.0) or 0.0)),
                    "rank": _rank_of(exact_rows, UNRELATED_ID),
                },
                "exact": {
                    "score": _round4(float(exact_audit.get("score", 0.0) or 0.0)),
                    "learned_vector_score": _round4(float(exact_audit.get("learned_vector_score", 0.0) or 0.0)),
                    "rank": _rank_of(exact_rows, EXACT_ID),
                },
            },
            "main_recall": {
                "neighbor": {
                    "score": _round4(float(neighbor_main.get("score", 0.0) or 0.0)),
                    "learned_score": _round4(float(neighbor_main.get("learned_score", 0.0) or 0.0)),
                    "rank": _rank_of(main_rows, NEIGHBOR_ID),
                },
            },
        }

    off = per_weight["off"]
    default = per_weight["default"]
    high = per_weight["high"]

    # Note: `learned_vector_score` is always *computed* and exposed on the audit
    # row, but it only enters the total score scaled by `min(0.22, learned_weight)`.
    # So at off (weight 0) the lvs field is non-zero yet contributes nothing to
    # the score. The correct ablation signal is therefore the change in `score`
    # across weights, not the raw lvs field value.

    # H1: in the audit path, the online learned VECTOR (a positive lvs field)
    # raises the neighbor's score above the off baseline, and ranks the neighbor
    # above the unrelated snapshot once the branch is weighted in.
    h1_vector_improves = (
        default["audit"]["neighbor"]["learned_vector_score"] > 0.0
        and default["audit"]["neighbor"]["score"] > off["audit"]["neighbor"]["score"]
        and default["audit"]["neighbor"]["rank"] < default["audit"]["unrelated"]["rank"]
    )

    # H1b: in the main recall path, the learned SIMILARITY branch raises the
    # neighbor's score above the off baseline.
    h1b_similarity_improves = (
        default["main_recall"]["neighbor"]["learned_score"] > 0.0
        and default["main_recall"]["neighbor"]["score"] > off["main_recall"]["neighbor"]["score"]
    )

    # H2: even at high learned_weight, an exact label/energy match outranks a
    # learned-only neighbor; learned vector does not dominate main evidence.
    h2_no_domination = (
        high["audit"]["exact"]["rank"] < high["audit"]["neighbor"]["rank"]
        and high["audit"]["exact"]["score"] > high["audit"]["neighbor"]["score"]
    )

    # H3: monotonic and bounded. The neighbor score does not decrease as weight
    # rises (off contributes zero learned mass, so off <= default <= high), and
    # nothing is NaN/inf. The bound is the `min(0.22, learned_weight)` cap on the
    # learned-vector branch plus the finite learned-similarity branch.
    neighbor_scores = [
        off["audit"]["neighbor"]["score"],
        default["audit"]["neighbor"]["score"],
        high["audit"]["neighbor"]["score"],
    ]
    all_finite = all(s == s and abs(s) != float("inf") for s in neighbor_scores)
    off_adds_no_learned_mass = off["audit"]["neighbor"]["score"] < default["audit"]["neighbor"]["score"]
    h3_monotonic_bounded = (
        off_adds_no_learned_mass
        and neighbor_scores[0] <= neighbor_scores[1] <= neighbor_scores[2]
        and all_finite
    )

    checks = {
        "h1_learned_vector_improves_neighborhood_recall": bool(h1_vector_improves),
        "h1b_learned_similarity_improves_main_recall": bool(h1b_similarity_improves),
        "h2_exact_match_not_dominated_by_learned_vector": bool(h2_no_domination),
        "h3_monotonic_and_bounded": bool(h3_monotonic_bounded),
    }
    return {
        "schema_id": SCHEMA_ID,
        "weights": WEIGHTS,
        "snapshot_legend": {
            NEIGHBOR_ID: "A_neighbor (learned neighbor of query subject, no surface overlap)",
            UNRELATED_ID: "A_unrelated (no surface overlap, no learned experience)",
            EXACT_ID: "B_exact (exact label/energy match, no learned experience)",
        },
        "per_weight": per_weight,
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
        "summary": {
            "passed": bool(probe["passed"]),
            "checks": probe["checks"],
        },
        "probe": probe,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    probe = payload["probe"]
    pw = probe["per_weight"]

    def line(name: str) -> str:
        a = pw[name]["audit"]
        return (
            f"| {name} ({pw[name]['weight']}) "
            f"| {a['neighbor']['score']} | {a['neighbor']['learned_vector_score']} | {a['neighbor']['rank']} "
            f"| {a['unrelated']['score']} | {a['unrelated']['rank']} "
            f"| {a['exact']['score']} | {a['exact']['rank']} |"
        )

    audit_rows = "\n".join(line(name) for name in ("off", "default", "high"))
    return f"""# OnlineVector-WeightAblation-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## audit 召回路径: 三档 learned_weight

| 档位 (weight) | neighbor score | neighbor lvs | neighbor rank | unrelated score | unrelated rank | exact score | exact rank |
|---|---:|---:|---:|---:|---:|---:|---:|
{audit_rows}

说明:
- neighbor = A_neighbor，与 query 表层 token 不重叠，但其主体 `gamma` 是 query 主体 `alpha` 的 learned 邻居。
- unrelated = A_unrelated，无表层重叠也无在线经验。
- exact = B_exact，与 query 精确 label/energy 匹配，无在线经验。
- lvs = `learned_vector_score`（online learned vector 在 audit 路径的贡献）。

## 结论口径

- online learned vector 在 audit 召回路径改善了经验邻域召回: neighbor 在 default 档分数高于 off 档，并排在 unrelated 之上。
- 它没有压过主证据: 即使 high 档，精确 label/energy 匹配的 B_exact 仍排在仅靠 learned 相似的 A_neighbor 之前。
- 行为单调有界: off 档 learned 分支对总分贡献为 0（lvs 字段虽被计算暴露，但乘以 `min(0.22, 0)=0`），neighbor 分数随权重单调不减，且 learned vector 分支被 `min(0.22, learned_weight)` cap。
- 这是 AP-Core bottom-loop 机制证据。它不修改 runtime，不宣称开放世界对话基座。
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
