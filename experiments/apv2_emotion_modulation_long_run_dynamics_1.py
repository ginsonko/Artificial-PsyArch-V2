from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.emotion.emotion_modulator import EmotionModulator  # noqa: E402


ARTIFACT_ID = "apv2_emotion_modulation_long_run_dynamics_1"
SCHEMA_ID = "apv2_emotion_modulation_long_run_dynamics_1/v1"
CREATED_AT = "2026-06-13"

BASELINES = {"DA": 0.12, "ADR": 0.05, "OXY": 0.12, "SER": 0.18, "END": 0.10, "COR": 0.06, "NOV": 0.08, "FOC": 0.10}

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: emotion slow-quantities form interpretable, decaying, auditable dynamics",
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


def _channels(modulator: EmotionModulator) -> dict[str, float]:
    return {k: _round4(v) for k, v in modulator.state.channels.items()}


def _probe() -> dict[str, Any]:
    modulator = EmotionModulator()
    trajectory: list[dict[str, Any]] = []

    def record(phase: str) -> None:
        mod = modulator.get_modulation()
        trajectory.append(
            {
                "phase": phase,
                "channels": _channels(modulator),
                "learning_rate_multiplier": _round4(mod["hdb"]["learning_rate_multiplier"]),
                "action_threshold_adjustment": _round4(mod["action"]["threshold_adjustment"]),
                "attention_resource_multiplier": _round4(mod["attention"]["resource_multiplier"]),
            }
        )

    # Phase 1: reward.
    for _ in range(5):
        modulator.update(cognitive_feelings={"channels": {}}, reward=0.6, punishment=0.0)
    record("reward")
    reward_state = trajectory[-1]

    # Phase 2: pressure / punishment.
    for _ in range(5):
        modulator.update(cognitive_feelings={"channels": {"pressure": 0.8, "dissonance": 0.6}}, reward=0.0, punishment=0.6)
    record("stress")
    stress_state = trajectory[-1]

    # Phase 3: silence (decay back to baseline).
    for _ in range(10):
        modulator.update(cognitive_feelings={"channels": {}}, reward=0.0, punishment=0.0)
    record("silence")
    silence_state = trajectory[-1]

    checks = {
        "reward_raises_da_and_learning_rate": reward_state["channels"]["DA"] > BASELINES["DA"]
        and reward_state["learning_rate_multiplier"] > 1.0,
        "stress_raises_cor_and_lowers_da": stress_state["channels"]["COR"] > BASELINES["COR"]
        and stress_state["channels"]["DA"] < reward_state["channels"]["DA"],
        "stress_lowers_action_threshold": stress_state["action_threshold_adjustment"] < 0.0,
        "silence_decays_toward_baseline": abs(silence_state["channels"]["DA"] - BASELINES["DA"])
        < abs(reward_state["channels"]["DA"] - BASELINES["DA"])
        and silence_state["channels"]["COR"] < stress_state["channels"]["COR"],
    }
    return {
        "schema_id": "emotion_modulation_long_run_probe/v1",
        "baselines": BASELINES,
        "trajectory": trajectory,
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
    rows = "\n".join(
        f"| {t['phase']} | {t['channels']['DA']} | {t['channels']['COR']} | {t['channels']['ADR']} | "
        f"{t['learning_rate_multiplier']} | {t['action_threshold_adjustment']} |"
        for t in p["trajectory"]
    )
    return f"""# EmotionModulation-LongRunDynamics-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 三阶段长跑 (baseline: DA={BASELINES['DA']}, COR={BASELINES['COR']})

| 阶段 | DA | COR | ADR | 学习率倍数 | 行动阈值调整 |
|---|---:|---:|---:|---:|---:|
{rows}

## 结论口径

- 奖励期: DA 升到 {p['trajectory'][0]['channels']['DA']}, 学习率倍数升到 {p['trajectory'][0]['learning_rate_multiplier']} —— 奖励驱动学习。
- 压力期: COR 升到 {p['trajectory'][1]['channels']['COR']}、ADR 升到 {p['trajectory'][1]['channels']['ADR']}, 行动阈值降到 {p['trajectory'][1]['action_threshold_adjustment']}(更警觉、更易行动), 同时惩罚把 DA 压回 {p['trajectory'][1]['channels']['DA']}。
- 静默期: DA 衰减回 {p['trajectory'][2]['channels']['DA']}(向 baseline 收敛), COR 从压力峰回落到 {p['trajectory'][2]['channels']['COR']}。
- 情绪慢量形成可解释、可衰减、可审计的慢动力学, 调制学习率/注意/行动阈值, 但不替代认知本身。
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
