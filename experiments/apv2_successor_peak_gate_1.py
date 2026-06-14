from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.short_term.focus_successor_bias import FocusSuccessorBias  # noqa: E402
from memory.spacetime.transition_store import TransitionStore  # noqa: E402


ARTIFACT_ID = "apv2_successor_peak_gate_1"
SCHEMA_ID = "apv2_successor_peak_gate_1/v1"
CREATED_AT = "2026-06-13"

BOUNDARY = {
    "artifact_id": ARTIFACT_ID,
    "route": "AP-Core bottom-loop mechanism hardening: successor-peak gating is process-shaped, not keyword-routed",
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
    return {"sa_label": label, "real_energy": float(real)}


def _gate_probe() -> dict[str, Any]:
    bias = FocusSuccessorBias()
    tick = 0

    # Sharp single peak: ctx_sharp is almost always followed by B_sharp (low entropy).
    for _ in range(12):
        bias.observe_transition(
            previous_focus_labels=["text::ctx_sharp"],
            current_focus_items=[_it("text::B_sharp")],
            tick_index=tick,
        )
        tick += 1

    # Blurry multi-peak: ctx_blur is followed by four different successors equally (high entropy).
    for label in ["text::W", "text::X", "text::Y", "text::Z"]:
        for _ in range(3):
            bias.observe_transition(
                previous_focus_labels=["text::ctx_blur"],
                current_focus_items=[_it(label)],
                tick_index=tick,
            )
            tick += 1

    sharp = bias.build_bias(
        previous_focus_labels=["text::ctx_sharp"],
        candidate_items=[_it("text::B_sharp")],
        tick_index=tick,
    )
    tick += 1
    blur = bias.build_bias(
        previous_focus_labels=["text::ctx_blur"],
        candidate_items=[_it("text::W"), _it("text::X"), _it("text::Y"), _it("text::Z")],
        tick_index=tick,
    )
    tick += 1

    sharp_bias = float(sharp["items"][0]["bias"]) if sharp["items"] else 0.0
    blur_top_bias = float(blur["items"][0]["bias"]) if blur["items"] else 0.0
    sharp_ctx = sharp["context_traces"][0] if sharp["context_traces"] else {}
    blur_ctx = blur["context_traces"][0] if blur["context_traces"] else {}
    sharp_entropy = float(sharp_ctx.get("branch_entropy", 0.0) or 0.0)
    blur_entropy = float(blur_ctx.get("branch_entropy", 0.0) or 0.0)
    sharp_damping = float(sharp_ctx.get("branch_damping", 0.0) or 0.0)
    blur_damping = float(blur_ctx.get("branch_damping", 0.0) or 0.0)
    entropy_floor = float(bias.entropy_floor)

    return {
        "schema_id": "successor_peak_gate_probe/v1",
        "sharp": {"top_bias": _round4(sharp_bias), "entropy": _round4(sharp_entropy), "damping": _round4(sharp_damping)},
        "blur": {"top_bias": _round4(blur_top_bias), "entropy": _round4(blur_entropy), "damping": _round4(blur_damping)},
        "entropy_floor": _round4(entropy_floor),
        "checks": {
            "sharp_peak_bias_exceeds_blur": sharp_bias > blur_top_bias,
            "sharp_entropy_below_blur": sharp_entropy < blur_entropy,
            "blur_damping_near_floor": abs(blur_damping - entropy_floor) < 0.02,
            "gate_is_soft_not_hard": blur_top_bias > 0.0,
        },
    }


def _lag_kernel_probe() -> dict[str, Any]:
    store = TransitionStore()
    k1 = store._lag_kernel(1)
    k2 = store._lag_kernel(2)
    k3 = store._lag_kernel(3)
    k4 = store._lag_kernel(4)
    return {
        "schema_id": "successor_lag_kernel_probe/v1",
        "lag_kernel": {"lag1": _round4(k1), "lag2": _round4(k2), "lag3": _round4(k3), "lag4": _round4(k4)},
        "checks": {
            "lag1_is_peak": k1 > k2,
            "lag1_to_lag2_is_cliff": (k1 / k2) > 2.0 if k2 > 0 else False,
            "tail_decays": k2 > k3 > k4,
            "tail_stays_positive": k4 > 0.0,
        },
    }


def _build_payload() -> dict[str, Any]:
    gate = _gate_probe()
    lag = _lag_kernel_probe()
    checks = {
        "gate_probe_passed": all(gate["checks"].values()),
        "lag_kernel_probe_passed": all(lag["checks"].values()),
    }
    return {
        "schema_id": SCHEMA_ID,
        "artifact_id": ARTIFACT_ID,
        "created_at": CREATED_AT,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": BOUNDARY,
        "summary": {"passed": all(checks.values()), "checks": checks},
        "gate_probe": gate,
        "lag_kernel_probe": lag,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    g = payload["gate_probe"]
    l = payload["lag_kernel_probe"]["lag_kernel"]
    return f"""# SuccessorPeakGate-1 报告

结论: {'通过' if payload['summary']['passed'] else '未通过'}

## 关键检查

```json
{json.dumps(payload['summary']['checks'], ensure_ascii=False, indent=2)}
```

## 峰型门控 (后继偏置由分布熵调制)

| context | 最强后继偏置 | branch_entropy | branch_damping |
|---|---:|---:|---:|
| 清晰单峰 ctx_sharp | {g['sharp']['top_bias']} | {g['sharp']['entropy']} | {g['sharp']['damping']} |
| 模糊多峰 ctx_blur | {g['blur']['top_bias']} | {g['blur']['entropy']} | {g['blur']['damping']} |

entropy_floor = {g['entropy_floor']}

## 后继 lag kernel 形状

| lag | kernel |
|---|---:|
| 1 | {l['lag1']} |
| 2 | {l['lag2']} |
| 3 | {l['lag3']} |
| 4 | {l['lag4']} |

## 结论口径

- 清晰单峰 context 给出强续写偏置 ({g['sharp']['top_bias']}), 模糊多峰 context 的偏置被压到接近 entropy_floor ({g['blur']['top_bias']}): 门控由后继分布形状(熵)决定, 不是关键词硬路由。
- 后继 lag kernel 是下一拍主峰 ({l['lag1']}) + 急降 ({l['lag2']}) + 长尾 ({l['lag3']}, {l['lag4']}) 的时间形状。
- 多峰偏置被压低但不为 0: 这是软门控, 不是硬关断。
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
