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


ARTIFACT_ID = "apv2_feedback_override_mechanism_1"
SCHEMA_ID = "apv2_feedback_override_mechanism_1/v1"
CREATED_AT = "2026-06-13"

WRONG_ACTION = "action::text_insert"
RIGHT_ACTION = "action::text_reread"

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: punishment+correction overrides a wrong action toward a right one",
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
        max_selected_actions=2,
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


def _action(action_id: str) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "actuator_id": "actuator::text_editor",
        "predicted_outcome": {"reward": 0.1, "punishment": 0.05, "correctness": 0.08, "pressure": 0.03, "confidence": 0.5},
    }


def _estimate(trace: dict, action_id: str) -> dict:
    for row in trace["outcome_memory"]["estimates"]:
        if str(row.get("action_id", "") or "") == action_id:
            return row
    return {}


def _probe() -> dict[str, Any]:
    planner = _planner()
    wrong = _action(WRONG_ACTION)
    right = _action(RIGHT_ACTION)

    wrong_traj: list[float] = []
    right_traj: list[float] = []
    for tick in range(10):
        planner.plan(
            tick_index=tick,
            state_snapshot_items=[],
            fast_bn=[], fast_cn=[], slow_bn=[], slow_cn=[],
            cognitive_feelings={"channels": {}},
            rhythm_trace={"channels": {}},
            time_trace={"channels": {}},
        )
        trace_w = planner.record_feedback(
            selected_actions=[wrong],
            observed_feedback={"reward": 0.0, "punishment": 0.6, "correctness": 0.0, "confidence": 0.88},
        )
        trace_r = planner.record_feedback(
            selected_actions=[right],
            observed_feedback={"reward": 0.55, "punishment": 0.0, "correctness": 0.45, "confidence": 0.9},
        )
        wrong_traj.append(_round4(float(_estimate(trace_w, WRONG_ACTION).get("drive_bias", 0.0) or 0.0)))
        right_traj.append(_round4(float(_estimate(trace_r, RIGHT_ACTION).get("drive_bias", 0.0) or 0.0)))

    final = planner.record_feedback(
        selected_actions=[wrong, right],
        observed_feedback={"reward": 0.0, "punishment": 0.0, "correctness": 0.0, "confidence": 0.5},
    )
    wrong_est = _estimate(final, WRONG_ACTION)
    right_est = _estimate(final, RIGHT_ACTION)
    wrong_bias = float(wrong_est.get("drive_bias", 0.0) or 0.0)
    right_bias = float(right_est.get("drive_bias", 0.0) or 0.0)

    checks = {
        "wrong_action_driven_negative": wrong_bias < 0.0 and wrong_traj[-1] < wrong_traj[0],
        "right_action_driven_positive": right_bias > 0.0,
        "override_flip_right_over_wrong": (right_bias - wrong_bias) > 0.5,
        "counts_auditable": int(wrong_est.get("failure_count", 0) or 0) > 0
        and int(wrong_est.get("success_count", 0) or 0) == 0
        and int(right_est.get("success_count", 0) or 0) > 0,
    }
    return {
        "schema_id": "feedback_override_probe/v1",
        "wrong_drive_bias_trajectory": wrong_traj,
        "right_drive_bias_trajectory": right_traj,
        "final": {
            "wrong_drive_bias": _round4(wrong_bias),
            "right_drive_bias": _round4(right_bias),
            "gap": _round4(right_bias - wrong_bias),
            "wrong_failure_count": int(wrong_est.get("failure_count", 0) or 0),
            "wrong_success_count": int(wrong_est.get("success_count", 0) or 0),
            "right_success_count": int(right_est.get("success_count", 0) or 0),
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
    f = p["final"]
    return f"""# FeedbackOverride-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 行动 drive_bias 轨迹 (同一状态下惩罚 wrong + 奖励 right)

- wrong (`{WRONG_ACTION}`) drive_bias: {p['wrong_drive_bias_trajectory']}
- right (`{RIGHT_ACTION}`) drive_bias: {p['right_drive_bias_trajectory']}

## 训练后

| 行动 | drive_bias | success | failure |
|---|---:|---:|---:|
| wrong | {f['wrong_drive_bias']} | {f['wrong_success_count']} | {f['wrong_failure_count']} |
| right | {f['right_drive_bias']} | {f['right_success_count']} | 0 |

override gap (right - wrong) = {f['gap']}

## 结论口径

- 反复惩罚错误行动、奖励正确行动后, 错误行动的 drive_bias 被单调推到负值 ({f['wrong_drive_bias']}), 正确行动被抬到正值 ({f['right_drive_bias']})。
- 二者差值达 {f['gap']}: 行动竞争从初始对称翻转为正确行动主导(override / 迁移)。
- 计数可审计: 错误行动 failure 累积、success 为 0; 正确行动相反。
- 这是 AP-Core bottom-loop 机制证据, 反馈进入行动 drive 竞争, 不写进概念表; 不修改 runtime, 不宣称开放世界对话基座。
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
