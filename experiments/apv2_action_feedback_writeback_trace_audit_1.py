from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime import APV21Runtime  # noqa: E402
from memory.store import MemoryStore  # noqa: E402


ARTIFACT_ID = "apv2_action_feedback_writeback_trace_audit_1"
SCHEMA_ID = "apv2_action_feedback_writeback_trace_audit_1/v1"
CREATED_AT = "2026-06-13"

REQUIRED_PROVENANCE = ["action_id", "observed_feedback", "outcome_memory_estimate", "feedback_energy_semantics", "causal_window"]

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: action feedback writes back as auditable SA",
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


def _probe() -> dict[str, Any]:
    runtime = APV21Runtime()
    feedback_items = runtime._build_action_feedback_items(
        selected_actions=[
            {
                "action_id": "action::text_insert",
                "predicted_outcome": {"reward": 0.0, "punishment": 0.7, "correctness": 0.0, "pressure": 0.2},
                "consequence_estimate": {},
            }
        ],
        observed_feedback={"reward": 0.0, "punishment": 0.7, "correctness": 0.0, "confidence": 0.9},
        planner_feedback={"outcome_memory": {"estimates": [{"action_id": "action::text_insert", "support": 0.5, "drive_bias": -0.25}]}},
        causal_window={"schema_id": "action_causal_window/v1", "action_ids": ["action::text_insert"]},
    )
    item = feedback_items[0] if feedback_items else {}
    anchor_meta = dict(item.get("anchor_meta", {}) or {})
    semantics = dict(anchor_meta.get("feedback_energy_semantics", {}) or {})
    drive_bias = float(anchor_meta.get("outcome_memory_estimate", {}).get("drive_bias", 0.0) or 0.0)
    real_energy = float(item.get("real_energy", 0.0) or 0.0)
    virtual_energy = float(item.get("virtual_energy", 0.0) or 0.0)
    provenance_present = [key for key in REQUIRED_PROVENANCE if key in anchor_meta]

    # Write back to memory and recall.
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
        items=feedback_items,
        focus_labels=[str(item.get("sa_label", "") or "")],
        source_text="action feedback",
    )
    memory.process_pending_index_jobs(budget=16, max_ms=200.0, include_heavy=True)
    query = [
        {
            "sa_label": str(item.get("sa_label", "") or ""),
            "real_energy": 1.0,
            "virtual_energy": 0.0,
            "cognitive_pressure": 1.0,
            "display_text": "feedback",
            "family": str(item.get("family", "action_feedback") or "action_feedback"),
            "source_type": "query",
        }
    ]
    rows = memory.recall(query, memory_kind="state", top_k=5)
    top_score = float(rows[0].get("score", 0.0) or 0.0) if rows else 0.0

    checks = {
        "feedback_sa_built": bool(feedback_items) and str(item.get("sa_label", "")).startswith("action_feedback::"),
        "punishment_becomes_virtual_energy": real_energy == 0.0 and virtual_energy > 0.0,
        "negative_drive_bias_preserved": drive_bias < 0.0,
        "provenance_complete": set(REQUIRED_PROVENANCE) <= set(anchor_meta.keys()),
        "feedback_written_and_recallable": top_score > 0.0,
    }
    return {
        "schema_id": "action_feedback_writeback_trace_probe/v1",
        "feedback_sa_label": str(item.get("sa_label", "") or ""),
        "real_energy": _round4(real_energy),
        "virtual_energy": _round4(virtual_energy),
        "feedback_energy_meaning": str(semantics.get("meaning", "") or ""),
        "outcome_drive_bias": _round4(drive_bias),
        "provenance_present": provenance_present,
        "recall_top_score": _round4(top_score),
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
    return f"""# ActionFeedbackWriteback-TraceAudit-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 端到端审计链

| 环节 | 值 |
|---|---|
| 反馈 SA label | `{p['feedback_sa_label']}` |
| real_energy | {p['real_energy']} |
| virtual_energy | {p['virtual_energy']} |
| 能量语义 | {p['feedback_energy_meaning']} |
| outcome drive_bias | {p['outcome_drive_bias']} |
| provenance 字段 | {', '.join(p['provenance_present'])} |
| 写回后召回 top score | {p['recall_top_score']} |

## 结论口径

- 行动后果被构造成一等反馈 SA `{p['feedback_sa_label']}`: 惩罚转为虚能量 ({p['virtual_energy']}) 作为 drive shaping, real_energy 为 0。
- 负向 drive_bias ({p['outcome_drive_bias']}) 与完整 provenance(action_id / observed_feedback / outcome_memory_estimate / feedback_energy_semantics / causal_window)一并写入。
- 反馈 SA 写入 MemoryStore 后可被召回 (score {p['recall_top_score']}): 反馈不是旁路日志, 而是可追踪、可召回的认知材料。
- AP-Core bottom-loop 机制证据, 反馈以能量/drive 进入认知而非写进答案表; 不修改 runtime, 不宣称开放世界对话基座。
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
