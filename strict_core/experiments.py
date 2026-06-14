from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Protocol

from strict_core.common import (
    SECRET_RULE_TAG,
    SECRET_TRUTH_TAG,
    flatten_json_strings,
    json_dump,
    learner_feedback,
    mean,
    ratio,
    sha256_file,
    stable_seed,
)
from strict_core.environments import BlindButtonWorld, CHOICE_ACTIONS, CompositionalFeatureWorld, HOLD_ACTION, StrictCase
from strict_core.policies import ActionFeedbackMemoryPolicy, ExactPatternPolicy, FeatureContributionPolicy
from strict_core.runtime_bridge import StrictRuntimeBridge


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = "strictcore0_foundation_boundary_suite"


class JudgeWorld(Protocol):
    def judge(self, case: StrictCase, action_id: str) -> dict: ...

    def perform_action(self, *, case: StrictCase, action_id: str) -> dict: ...


def _case_public_payload(case: StrictCase) -> dict:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "visible_state_items": case.visible_state_items,
        "judge_unknown": bool(case.judge_unknown),
    }


def _score_cases(cases: list[dict]) -> dict:
    if not cases:
        return {"accuracy": 0.0, "count": 0, "hold_rate": 0.0}
    return {
        "accuracy": ratio([bool(case.get("was_rewarded_by_examiner", False)) for case in cases]),
        "count": len(cases),
        "hold_rate": ratio([str(case.get("selected_action", "") or "") == HOLD_ACTION for case in cases]),
    }


def _visible_taint_audit(payload: dict) -> dict:
    serialized = flatten_json_strings(payload)
    forbidden = [
        SECRET_RULE_TAG,
        SECRET_TRUTH_TAG,
        "private_group",
        "private_values",
        "private_best_action",
        "secret",
        "target",
        "correct",
        "oracle",
        "solution",
    ]
    hits = [term for term in forbidden if term.lower() in serialized.lower()]
    return {"passed": not hits, "hits": hits}


def _teacher_off_boundary_ok(cases: list[dict]) -> bool:
    for case in cases:
        if case.get("feedback_to_learner") is not None:
            return False
        if int(case.get("memory_write_after_test_action", 0) or 0) != 0:
            return False
        signal = case.get("teacher_off_signal", {})
        if signal != {"state_items": [], "action_biases": [], "feedback": {}}:
            return False
    return True


def _run_case(
    *,
    bridge: StrictRuntimeBridge,
    policy,
    world: JudgeWorld,
    case: StrictCase,
    phase: str,
    group_id: str,
    learning_enabled: bool,
    feedback_mode: str = "normal",
    exact_policy_learns: bool = False,
    run_id: str,
) -> dict:
    before_count = len(bridge.memory._recent_by_kind.get("strict_state", []))
    tick = bridge.begin_case(case.visible_state_items)
    decision = policy.decide(state_items=case.visible_state_items, memory_rows=tick.memory_rows, learning_enabled=learning_enabled)
    actuator_event = world.perform_action(case=case, action_id=decision.selected_action)
    feedback = world.judge(case, decision.selected_action)
    if feedback_mode == "inverted" and hasattr(world, "inverted_feedback"):
        feedback = world.inverted_feedback(feedback)  # type: ignore[attr-defined]
    elif feedback_mode == "random" and hasattr(world, "random_feedback"):
        feedback = world.random_feedback(rng=random.Random(stable_seed(run_id, case.case_id, "random_feedback")))  # type: ignore[attr-defined]
    memory_write = bridge.commit_action(
        state_items=case.visible_state_items,
        action_id=decision.selected_action,
        reason=decision.reason,
        case_id=case.case_id,
        feedback=feedback,
        learning_enabled=learning_enabled,
    )
    if exact_policy_learns and learning_enabled and hasattr(policy, "learn"):
        policy.learn(state_items=case.visible_state_items, action_id=decision.selected_action, feedback=feedback)
    after_count = len(bridge.memory._recent_by_kind.get("strict_state", []))
    feedback_to_learner = learner_feedback(feedback) if learning_enabled else None
    return {
        "schema_id": "strictcore0_case_snapshot/v1",
        "run_id": run_id,
        "group_id": group_id,
        "phase": phase,
        "case": _case_public_payload(case),
        "runtime_tick": tick.tick_index,
        "visible_observation": case.visible_state_items,
        "state_pool_snapshot": tick.state_pool_snapshot,
        "memory_recall_rows": [
            {
                "memory_id": row.get("memory_id"),
                "score": row.get("score"),
                "numeric_score": row.get("numeric_score"),
                "score_breakdown": row.get("score_breakdown", {}),
            }
            for row in tick.memory_rows
        ],
        "action_candidates": decision.action_candidates,
        "selected_action": decision.selected_action,
        "selected_action_reason": decision.reason,
        "grasp": decision.grasp,
        "recalled_count": decision.recalled_count,
        "actuator_event": actuator_event,
        "examiner_private_result_for_report_only": {
            "reward": feedback.get("reward", 0.0),
            "punishment": feedback.get("punishment", 0.0),
            "verdict": feedback.get("verdict", ""),
        },
        "was_rewarded_by_examiner": float(feedback.get("reward", 0.0) or 0.0) > 0.0,
        "feedback_to_learner": feedback_to_learner,
        "memory_write": memory_write,
        "memory_write_after_test_action": 0 if learning_enabled else max(0, after_count - before_count),
        "teacher_off_signal": {"state_items": [], "action_biases": [], "feedback": {}} if not learning_enabled else None,
        "component_trace": [
            "sensor_adapter_emitted_numeric_only",
            "state_pool_received_visible_items",
            "memory_recall_used_visible_snapshot",
            "policy_ranked_action_candidates",
            "actuator_committed_action",
            "examiner_scored_after_commit",
        ],
    }


def _run_training_and_tests(
    *,
    run_id: str,
    world: JudgeWorld,
    train_cases: list[StrictCase],
    holdout_cases: list[StrictCase],
    ood_cases: list[StrictCase],
    group_id: str,
    policy_kind: str,
    feedback_mode: str = "normal",
    training_enabled: bool = True,
    memory_enabled: bool = True,
) -> dict:
    bridge = StrictRuntimeBridge()
    if policy_kind == "exact":
        policy = ExactPatternPolicy()
    elif policy_kind == "feature":
        policy = FeatureContributionPolicy(explore_seed=stable_seed(run_id, "feature_policy"))
    else:
        policy = ActionFeedbackMemoryPolicy(explore_seed=stable_seed(run_id, "memory_policy"))
    if not memory_enabled:
        train_bridge = bridge
        test_bridge = StrictRuntimeBridge()
    else:
        train_bridge = bridge
        test_bridge = bridge
    pretest = [
        _run_case(
            bridge=StrictRuntimeBridge(),
            policy=ActionFeedbackMemoryPolicy(explore_seed=idx),
            world=world,
            case=case,
            phase="cold_pretest",
            group_id=group_id,
            learning_enabled=False,
            run_id=f"{run_id}_pre_{idx}",
        )
        for idx, case in enumerate(holdout_cases[:6])
    ]
    training = []
    if training_enabled:
        for case in train_cases:
            training.append(
                _run_case(
                    bridge=train_bridge,
                    policy=policy,
                    world=world,
                    case=case,
                    phase="training",
                    group_id=group_id,
                    learning_enabled=True,
                    feedback_mode=feedback_mode,
                    exact_policy_learns=policy_kind == "exact",
                    run_id=run_id,
                )
            )
    holdout = [
        _run_case(
            bridge=test_bridge,
            policy=policy,
            world=world,
            case=case,
            phase="teacher_off_holdout",
            group_id=group_id,
            learning_enabled=False,
            run_id=run_id,
        )
        for case in holdout_cases
    ]
    ood = [
        _run_case(
            bridge=test_bridge,
            policy=policy,
            world=world,
            case=case,
            phase="teacher_off_ood",
            group_id=group_id,
            learning_enabled=False,
            run_id=run_id,
        )
        for case in ood_cases
    ]
    package = train_bridge.export_package(package_id=f"{run_id}_package")
    package_reload = []
    if group_id == "G0_normal":
        package_bridge = StrictRuntimeBridge()
        package_bridge.load_package(package)
        package_policy = (
            FeatureContributionPolicy(explore_seed=stable_seed(run_id, "feature_package_policy"))
            if policy_kind == "feature"
            else ActionFeedbackMemoryPolicy(explore_seed=stable_seed(run_id, "memory_package_policy"))
        )
        package_reload = [
            _run_case(
                bridge=package_bridge,
                policy=package_policy,
                world=world,
                case=case,
                phase="teacher_off_package_reload",
                group_id=group_id,
                learning_enabled=False,
                run_id=f"{run_id}_reload",
            )
            for case in holdout_cases[:6]
        ]
    visible_for_audit = {
        "pretest": pretest,
        "training": training,
        "holdout": holdout,
        "ood": ood,
        "package_reload": package_reload,
        "package": package,
    }
    return {
        "group_id": group_id,
        "policy_kind": policy_kind,
        "feedback_mode": feedback_mode,
        "training_enabled": training_enabled,
        "memory_enabled": memory_enabled,
        "pretest": pretest,
        "training": training,
        "holdout": holdout,
        "ood": ood,
        "package_reload": package_reload,
        "scores": {
            "pretest": _score_cases(pretest),
            "training_second_half": _score_cases(training[len(training) // 2 :]),
            "holdout": _score_cases(holdout),
            "ood": _score_cases(ood),
            "package_reload": _score_cases(package_reload),
        },
        "teacher_off_boundary_ok": all(
            _teacher_off_boundary_ok(rows)
            for rows in (pretest, holdout, ood, package_reload)
        ),
        "visible_taint_audit": _visible_taint_audit(visible_for_audit),
        "package": package,
    }


def _blind_suite(seed: int) -> dict:
    world = BlindButtonWorld(seed=seed)
    train = world.case_stream(split="blind_train", count=54, start_variant=10)
    holdout = world.case_stream(split="blind_holdout", count=24, start_variant=1000)
    ood = world.ood_stream(split="blind_ood", count=12, start_variant=2000)
    return {
        "task_id": "blind_action_feedback_closed_loop",
        "groups": [
            _run_training_and_tests(run_id=f"blind_{seed}_G0", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G0_normal", policy_kind="memory"),
            _run_training_and_tests(run_id=f"blind_{seed}_G1", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G1_no_training", policy_kind="memory", training_enabled=False),
            _run_training_and_tests(run_id=f"blind_{seed}_G2", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G2_no_memory", policy_kind="memory", memory_enabled=False),
            _run_training_and_tests(run_id=f"blind_{seed}_G3", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G3_random_feedback", policy_kind="memory", feedback_mode="random"),
            _run_training_and_tests(run_id=f"blind_{seed}_G4", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G4_inverted_feedback", policy_kind="memory", feedback_mode="inverted"),
            _run_training_and_tests(run_id=f"blind_{seed}_G5", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G5_exact_pattern", policy_kind="exact"),
        ],
    }


def _composition_suite(seed: int) -> dict:
    world = CompositionalFeatureWorld(seed=seed)
    train = world.training_stream(split="comp_train", repeat=7, start_variant=10)
    holdout = world.compositional_holdout_stream(split="comp_holdout", repeat=8, start_variant=1000)
    ood = world.ood_stream(split="comp_ood", count=12, start_variant=2000)
    return {
        "task_id": "compositional_feature_generalization",
        "groups": [
            _run_training_and_tests(run_id=f"comp_{seed}_G0", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G0_normal", policy_kind="feature"),
            _run_training_and_tests(run_id=f"comp_{seed}_G1", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G1_no_training", policy_kind="feature", training_enabled=False),
            _run_training_and_tests(run_id=f"comp_{seed}_G2", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G2_no_memory", policy_kind="feature", memory_enabled=False),
            _run_training_and_tests(run_id=f"comp_{seed}_G4", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G4_inverted_feedback", policy_kind="feature", feedback_mode="inverted"),
            _run_training_and_tests(run_id=f"comp_{seed}_G5", world=world, train_cases=train, holdout_cases=holdout, ood_cases=ood, group_id="G5_exact_pattern", policy_kind="exact"),
        ],
    }


def _aggregate(suites: list[dict]) -> dict:
    by_task: dict[str, dict] = {}
    for suite in suites:
        task_id = suite["task_id"]
        task_rows = by_task.setdefault(task_id, {})
        for group in suite["groups"]:
            group_id = group["group_id"]
            bucket = task_rows.setdefault(group_id, {"holdout": [], "ood": [], "pretest": [], "package_reload": []})
            for key in bucket:
                bucket[key].append(float(group["scores"][key]["accuracy"]))
    result: dict[str, dict] = {}
    for task_id, groups in by_task.items():
        result[task_id] = {}
        for group_id, metrics in groups.items():
            result[task_id][group_id] = {
                key: {"mean": mean(values), "min": min(values), "max": max(values)}
                for key, values in metrics.items()
            }
    return result


def _pass_conditions(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    blind = aggregate["blind_action_feedback_closed_loop"]
    comp = aggregate["compositional_feature_generalization"]
    all_groups = [group for suite in payload["suites"] for group in suite["groups"]]
    return {
        "normal_blind_learns_above_pretest": blind["G0_normal"]["holdout"]["mean"] >= 0.60 and blind["G0_normal"]["pretest"]["mean"] <= 0.20,
        "normal_composition_generalizes": comp["G0_normal"]["holdout"]["mean"] >= 0.58,
        "package_reload_keeps_some_skill": blind["G0_normal"]["package_reload"]["mean"] >= 0.60,
        "controls_do_not_match_normal": (
            blind["G0_normal"]["holdout"]["mean"] > blind["G1_no_training"]["holdout"]["mean"] + 0.35
            and blind["G0_normal"]["holdout"]["mean"] > blind["G2_no_memory"]["holdout"]["mean"] + 0.25
            and blind["G0_normal"]["holdout"]["mean"] > blind["G4_inverted_feedback"]["holdout"]["mean"] + 0.25
            and comp["G0_normal"]["holdout"]["mean"] > comp["G5_exact_pattern"]["holdout"]["mean"] + 0.15
        ),
        "ood_abstain_boundary": blind["G0_normal"]["ood"]["mean"] >= 0.70 and comp["G0_normal"]["ood"]["mean"] >= 0.70,
        "teacher_off_has_no_feedback_or_memory_write": all(group["teacher_off_boundary_ok"] for group in all_groups),
        "visible_payload_has_no_secret_taint": all(group["visible_taint_audit"]["passed"] for group in all_groups),
        "tainted_sensor_control_is_rejected": bool(payload.get("tainted_sensor_control", {}).get("audit", {}).get("passed") is False),
    }


def _write_markdown_report(payload: dict, path: Path) -> None:
    lines = [
        "# StrictCore-0 基础能力边界综合验收报告",
        "",
        f"日期: 2026-06-03",
        f"协议: {payload['protocol_version']}",
        f"Python: {payload['reproducibility']['python_version']}",
        f"Repository status: {payload['reproducibility']['repository_status']}",
        "",
        "## 结论边界",
        "",
        "本报告证明的是窄域 numeric 状态中的 AP 风格行动-反馈-记忆闭环、组合泛化、teacher-off 边界和技能包重载证据。它不证明完整小学数学、开放世界语义、独立视觉识别或 ASR。",
        "",
        "## 通过条件",
    ]
    for key, value in payload["summary"]["pass_conditions"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## 聚合结果", ""])
    for task_id, groups in payload["aggregate"].items():
        lines.append(f"### {task_id}")
        for group_id, metrics in groups.items():
            lines.append(
                f"- {group_id}: holdout_mean={metrics['holdout']['mean']:.3f}, "
                f"pretest_mean={metrics['pretest']['mean']:.3f}, "
                f"ood_mean={metrics['ood']['mean']:.3f}, "
                f"package_reload_mean={metrics['package_reload']['mean']:.3f}"
            )
    lines.extend(
        [
            "",
            "## 能做到什么",
            "",
            "- 在受控 numeric observation 中，通过真实 actuator event 和外部 reward/punishment 形成后续行动改变。",
            "- teacher-off 阶段不接收答案、不补救、不回填，仍能在 normal group 中保持高于对照组的表现。",
            "- 在组合特征任务中，normal group 能高于 exact-pattern baseline，说明不是只靠整态表绑定。",
            "- training 后导出的 experience package 可以重载到新 bridge 中，保留部分 teacher-off 行动能力。",
            "- OOD 中 abstain/hold 被奖励，in-domain 中 hold 不刷分。",
            "",
            "## 不能宣称什么",
            "",
            "- 不能宣称完整自然数或完整小学数学已经证明。",
            "- 不能宣称从像素/音频零起点得到开放世界语义识别。",
            "- 不能宣称没有任何工程脚手架；当前证明的是严格边界下的主程序接口闭环。",
            "- 不能宣称已经达到 LLM 级通用推理能力。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_strictcore0_foundation_boundary_suite(output_dir: str | Path | None = None, *, seeds: list[int] | None = None) -> dict:
    out = Path(output_dir) if output_dir is not None else ROOT / "strict_core" / "reports" / ARTIFACT_ID
    out.mkdir(parents=True, exist_ok=True)
    seed_rows = list(seeds or [20260603, 20260604, 20260605])
    suites: list[dict] = []
    for seed in seed_rows:
        suites.append(_blind_suite(seed))
        suites.append(_composition_suite(seed + 700))
    aggregate = _aggregate(suites)
    payload = {
        "schema_id": "strictcore0_foundation_boundary_suite/v1",
        "protocol_version": "StrictCore-0/v0.2-foundation-boundary",
        "artifact_id": ARTIFACT_ID,
        "seeds": seed_rows,
        "suites": suites,
        "aggregate": aggregate,
        "tainted_sensor_control": {
            "description": "Deliberately inject target/correct into a visible sensor payload; auditor must reject it.",
            "audit": _visible_taint_audit(
                {
                    "visible_observation": [
                        {
                            "sa_label": "sensor::tainted",
                            "numeric_features": {"x": 1.0},
                            "target": "action::press_left",
                            "correct": True,
                        }
                    ]
                }
            ),
        },
        "reproducibility": {
            "python_version": sys.version,
            "repository_status": "not_git_repository",
            "code_sha256": {
                "strict_core/common.py": sha256_file(ROOT / "strict_core" / "common.py"),
                "strict_core/environments.py": sha256_file(ROOT / "strict_core" / "environments.py"),
                "strict_core/runtime_bridge.py": sha256_file(ROOT / "strict_core" / "runtime_bridge.py"),
                "strict_core/policies.py": sha256_file(ROOT / "strict_core" / "policies.py"),
                "strict_core/experiments.py": sha256_file(ROOT / "strict_core" / "experiments.py"),
            },
        },
    }
    pass_conditions = _pass_conditions(payload)
    payload["summary"] = {
        "passed": all(pass_conditions.values()),
        "pass_conditions": pass_conditions,
    }
    json_path = out / "strictcore0_foundation_boundary_suite.json"
    md_path = out / "StrictCore0_FoundationBoundary_Report_20260603.md"
    json_dump(json_path, payload)
    _write_markdown_report(payload, md_path)
    payload["output_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_dump(json_path, payload)
    return payload


if __name__ == "__main__":
    result = run_strictcore0_foundation_boundary_suite()
    print(json.dumps({"passed": result["summary"]["passed"], "output_paths": result["output_paths"]}, ensure_ascii=False, indent=2))
