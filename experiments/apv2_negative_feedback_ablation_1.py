from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.action import ActionConsequencePlanner  # noqa: E402


ARTIFACT_ID = "apv2_negative_feedback_ablation_1"
SCHEMA_ID = "apv2_negative_feedback_ablation_1/v1"
CREATED_AT = "2026-06-13"

ACTION_ID = "action::inspect_residual"

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: ablating negative feedback removes error suppression",
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


def _planner() -> ActionConsequencePlanner:
    return ActionConsequencePlanner(
        enabled=True,
        selection_threshold=0.1,
        max_selected_actions=1,
        fatigue_decay=0.92,
        fatigue_step=0.0,
        bias_learning_rate=0.0,
        bias_gain=0.0,
        confidence_gain=0.18,
        wait_base_drive=0.18,
        outcome_memory_enabled=True,
        outcome_memory_learning_rate=0.22,
        outcome_memory_decay_per_tick=0.998,
        outcome_memory_support_scale=2.0,
        outcome_memory_max_drive_bias=0.75,
    )


def _action() -> dict[str, Any]:
    return {
        "action_id": ACTION_ID,
        "actuator_id": "actuator::attention",
        "predicted_outcome": {"reward": 0.1, "punishment": 0.05, "correctness": 0.08, "pressure": 0.03, "confidence": 0.5},
    }


def _run_condition(observed_feedback: dict[str, float]) -> dict[str, Any]:
    planner = _planner()
    action = _action()
    traj: list[float] = []
    for tick in range(10):
        planner.plan(
            tick_index=tick,
            state_snapshot_items=[],
            fast_bn=[], fast_cn=[], slow_bn=[], slow_cn=[],
            cognitive_feelings={"channels": {}},
            rhythm_trace={"channels": {}},
            time_trace={"channels": {}},
        )
        trace = planner.record_feedback(selected_actions=[action], observed_feedback=observed_feedback)
        estimate = next(r for r in trace["outcome_memory"]["estimates"] if r["action_id"] == ACTION_ID)
        traj.append(_round4(float(estimate.get("drive_bias", 0.0) or 0.0)))
    return {"trajectory": traj, "final": traj[-1]}


def _probe() -> dict[str, Any]:
    full = _run_condition({"reward": 0.0, "punishment": 0.6, "correctness": 0.0, "confidence": 0.88})
    no_negative = _run_condition({"reward": 0.0, "punishment": 0.0, "correctness": 0.0, "confidence": 0.5})
    positive_control = _run_condition({"reward": 0.5, "punishment": 0.0, "correctness": 0.4, "confidence": 0.9})

    full_final = float(full["final"])
    no_neg_final = float(no_negative["final"])
    pos_final = float(positive_control["final"])

    checks = {
        "negative_feedback_suppresses_error": full_final < 0.0,
        "ablation_removes_suppression": (no_neg_final - full_final) > 0.2 and abs(no_neg_final) < 0.05,
        "outcome_memory_is_not_dead": pos_final > 0.0,
        "full_trajectory_monotone_down": all(
            full["trajectory"][i + 1] <= full["trajectory"][i] + 1e-9 for i in range(len(full["trajectory"]) - 1)
        ),
    }
    return {
        "schema_id": "negative_feedback_ablation_probe/v1",
        "full": full,
        "no_negative": no_negative,
        "positive_control": positive_control,
        "summary_finals": {
            "full": _round4(full_final),
            "no_negative": _round4(no_neg_final),
            "positive_control": _round4(pos_final),
        },
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
    s = p["summary_finals"]
    return f"""# NegativeFeedback-Ablation-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 三条件下错误行动的最终 drive_bias

| 条件 | 反馈 | 最终 drive_bias |
|---|---|---:|
| full (负反馈在场) | punishment=0.6 | {s['full']} |
| no_negative (消融负反馈) | 中性 | {s['no_negative']} |
| positive_control (正反馈) | reward=0.5 | {s['positive_control']} |

full 轨迹: {p['full']['trajectory']}

## 结论口径

- 有负反馈时, 错误行动 drive_bias 被抑制为 {s['full']}; 去掉负反馈信号后, 同一错误行动 drive_bias 停在 {s['no_negative']}(几乎不被抑制)。
- 二者差距说明: 负反馈对"抑制错误行动"有独立因果贡献, 不是 UI 文案。
- 正反馈对照达到 {s['positive_control']}, 说明 outcome memory 能双向移动, 消融效果不是因为机制坏了。
- 这是 AP-Core bottom-loop 机制证据, 消融的是反馈信号而非机制代码; 不修改 runtime, 不宣称开放世界对话基座。
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
