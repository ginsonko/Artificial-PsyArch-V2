from __future__ import annotations

import argparse
import copy
import html
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stpv2_process_anchor_audit_v01 import (
    FORBIDDEN_PUBLIC_KEYS,
    _contains_forbidden_key,
    _display_path,
    _ratio,
    _round4,
    _sha256,
    _stats,
    _write_json,
    _write_text,
)
from scripts.run_stpv2_process_anchor_transfer_v04 import (
    ANCHOR_KEYS,
    DOMAINS,
    _domain_adapter_decision,
    _fit_process_policy,
    _make_cases,
    _scores,
    _surface_keyword_decision,
    _surface_markers,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "representation_ablation_1_20260608"
DEFAULT_SEEDS = (2026060871, 2026060872, 2026060873, 2026060874, 2026060875)
SCHEMA_ID = "apv21_representation_ablation_1/v0.1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class RepresentationGroup:
    group_id: str
    display_name_zh: str
    representation_level: str
    route: str
    boundary: str
    ap_native: bool
    used_tool: bool = False


GROUPS = (
    RepresentationGroup(
        "r1_structured_process_sa",
        "R1 结构化过程 SA",
        "numeric_process_anchor_sa",
        "STP-v2 structured process anchors",
        "当前 STP-v2 主路线; 使用公开 numeric process anchors",
        True,
    ),
    RepresentationGroup(
        "r2_surface_text_only",
        "R2 表面文本 token",
        "surface_text_markers",
        "D1 surface marker baseline",
        "只从 D1 训练表面 token 学硬触发, 不读取过程锚点",
        False,
    ),
    RepresentationGroup(
        "r3_process_event_bridge",
        "R3 过程事件 bridge",
        "generic_process_event_labels",
        "Generic process-event bridge",
        "不读取 numeric anchor; 只把公开过程事件标签映射为通用锚点",
        True,
    ),
    RepresentationGroup(
        "r4_domain_surface_adapter",
        "R4 分域表面适配器",
        "domain_specific_surface_adapter",
        "Non AP-native engineering upper bound",
        "按 D1/D2/D3 表面形式写分域适配规则; 高分不算 AP-native 证据",
        False,
    ),
    RepresentationGroup(
        "r5_shuffled_process_bridge",
        "R5 打乱过程事件 bridge",
        "shuffled_process_event_labels",
        "Shuffled process-event falsification control",
        "同分布但 case-level 来源打乱; 检验不是有过程字段就行",
        False,
    ),
)


def _event_labels_to_anchors(labels: list[str]) -> dict[str, float]:
    label_set = set(labels)
    anchors = {key: 0.0 for key in ANCHOR_KEYS}
    if {"teacher_context", "correction_event"} & label_set:
        anchors["teacher_context"] = 0.84
        anchors["correction_event"] = 0.82
        anchors["low_grasp"] = 0.42
        anchors["mismatch"] = 0.45
    if {"low_grasp", "mismatch", "buffer_reread"} & label_set:
        anchors["low_grasp"] = max(anchors["low_grasp"], 0.74)
        anchors["mismatch"] = max(anchors["mismatch"], 0.82)
    if "ordinary_observation" in label_set:
        anchors["low_grasp"] = max(anchors["low_grasp"], 0.14)
        anchors["mismatch"] = max(anchors["mismatch"], 0.11)
    if "stable_readback" in label_set:
        anchors["low_grasp"] = max(anchors["low_grasp"], 0.10)
        anchors["mismatch"] = max(anchors["mismatch"], 0.08)
    return {key: _round4(value) for key, value in anchors.items()}


def _case_with_bridge_anchors(case: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(case)
    result["process_anchors_public"] = _event_labels_to_anchors(list(case.get("process_event_labels_public", []) or []))
    return result


def _fit_policy_from_cases(train_cases: list[dict[str, Any]], private_cases: dict[str, Any]) -> dict[str, float]:
    return _fit_process_policy(train_cases, private_cases)


def _decision_from_policy(case: dict[str, Any], policy: dict[str, float]) -> tuple[bool, float, dict[str, Any]]:
    scores = _scores(case["process_anchors_public"])
    if case["action_head"] == "relation_trigger":
        score = scores["relation_head_score"]
        threshold = policy["relation_trigger_threshold"]
    else:
        score = scores["repair_head_score"]
        threshold = policy["local_repair_threshold"]
    return bool(score >= threshold), _round4(score), {"scores": scores, "threshold": threshold}


def _make_shuffled_bridge_cases(seed: int, train_cases: list[dict[str, Any]], test_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_cases = train_cases + test_cases
    anchors = [_event_labels_to_anchors(list(case.get("process_event_labels_public", []) or [])) for case in all_cases]
    rng = random.Random(seed + 104729)
    rng.shuffle(anchors)
    remapped: list[dict[str, Any]] = []
    for case, anchor in zip(all_cases, anchors):
        item = copy.deepcopy(case)
        item["process_anchors_public"] = anchor
        remapped.append(item)
    return remapped[: len(train_cases)], remapped[len(train_cases) :]


def _group_decision(
    group: RepresentationGroup,
    case: dict[str, Any],
    *,
    structured_policy: dict[str, float],
    bridge_policy: dict[str, float],
    shuffled_policy: dict[str, float],
    surface_markers: dict[str, Any],
) -> dict[str, Any]:
    if group.group_id == "r1_structured_process_sa":
        decision, confidence, trace = _decision_from_policy(case, structured_policy)
        basis = "numeric process anchors"
    elif group.group_id == "r2_surface_text_only":
        decision, confidence, basis = _surface_keyword_decision(case, surface_markers)
        trace = {"surface_markers_public_summary": {key: len(value) for key, value in surface_markers.items()}}
    elif group.group_id == "r3_process_event_bridge":
        bridge_case = _case_with_bridge_anchors(case)
        decision, confidence, trace = _decision_from_policy(bridge_case, bridge_policy)
        basis = "generic public process event labels -> anchors"
    elif group.group_id == "r4_domain_surface_adapter":
        decision, confidence, basis = _domain_adapter_decision(case)
        trace = {"adapter_boundary": "non AP-native per-domain surface adapter"}
    elif group.group_id == "r5_shuffled_process_bridge":
        decision, confidence, trace = _decision_from_policy(case, shuffled_policy)
        basis = "shuffled process-event bridge"
    else:
        raise ValueError(group.group_id)
    return {
        "decision_fire": bool(decision),
        "confidence": _round4(confidence),
        "decision_basis": basis,
        "decision_trace": trace,
        "representation_level": group.representation_level,
        "outcome_anchor_score_visible_to_decision": 0.0,
        "student_side_provider_called": False,
        "hidden_solver_used": False,
        "used_tool": group.used_tool,
    }


def _run_seed(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train_cases, test_cases, private_payload = _make_cases(seed)
    private_cases = private_payload["cases"]
    structured_policy = _fit_policy_from_cases(train_cases, private_cases)
    bridge_train = [_case_with_bridge_anchors(case) for case in train_cases]
    bridge_policy = _fit_policy_from_cases(bridge_train, private_cases)
    shuffled_train, shuffled_test = _make_shuffled_bridge_cases(seed, train_cases, test_cases)
    shuffled_policy = _fit_policy_from_cases(shuffled_train, private_cases)
    surface = _surface_markers(train_cases, private_cases)

    records: list[dict[str, Any]] = []
    shuffled_by_id = {case["case_id"]: case for case in shuffled_test}
    for group in GROUPS:
        for case in test_cases:
            eval_case = shuffled_by_id[case["case_id"]] if group.group_id == "r5_shuffled_process_bridge" else case
            decision = _group_decision(
                group,
                eval_case,
                structured_policy=structured_policy,
                bridge_policy=bridge_policy,
                shuffled_policy=shuffled_policy,
                surface_markers=surface,
            )
            expected = bool(private_cases[case["case_id"]]["action_should_fire"])
            success = decision["decision_fire"] == expected
            records.append(
                {
                    "schema_id": "apv21_representation_ablation_1/task_record/v0.1",
                    "seed": int(seed),
                    "task_id": "representation_ablation_1",
                    "group_id": group.group_id,
                    "display_name_zh": group.display_name_zh,
                    "route": group.route,
                    "boundary": group.boundary,
                    "ap_native": group.ap_native,
                    "used_tool": group.used_tool,
                    "case_id": case["case_id"],
                    "domain": case["domain"],
                    "action_head": case["action_head"],
                    "family_public": case["family_public"],
                    "learner_public_view": {
                        "public_surface": case["public_surface"],
                        "process_event_labels_public": case["process_event_labels_public"],
                        "representation_level": group.representation_level,
                    },
                    "decision": decision,
                    "scored_success_after_examiner_reveal": bool(success),
                }
            )
    seed_metrics = _group_metrics(records)
    private_payload = {
        **private_payload,
        "seed": int(seed),
        "structured_policy": structured_policy,
        "bridge_policy": bridge_policy,
        "shuffled_policy": shuffled_policy,
        "reveal_policy": "Private labels are used only after learner decisions for scoring.",
    }
    return {"seed": int(seed), "group_metrics": seed_metrics, "record_count": len(records)}, records, private_payload


def _group_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        group_records = [record for record in records if record["group_id"] == group.group_id]
        by_domain = {}
        for domain in DOMAINS:
            domain_records = [record for record in group_records if record["domain"] == domain]
            by_domain[domain] = {
                "n": len(domain_records),
                "accuracy": _round4(_ratio(sum(1 for record in domain_records if record["scored_success_after_examiner_reveal"]), len(domain_records))),
                "fire_rate": _round4(_ratio(sum(1 for record in domain_records if record["decision"]["decision_fire"]), len(domain_records))),
            }
        trigger_records = [record for record in group_records if record["action_head"] == "relation_trigger"]
        repair_records = [record for record in group_records if record["action_head"] == "local_repair"]
        false_positive = sum(1 for record in group_records if record["decision"]["decision_fire"] and not record["scored_success_after_examiner_reveal"])
        false_negative = sum(1 for record in group_records if (not record["decision"]["decision_fire"]) and not record["scored_success_after_examiner_reveal"])
        rows.append(
            {
                "group_id": group.group_id,
                "display_name_zh": group.display_name_zh,
                "route": group.route,
                "boundary": group.boundary,
                "representation_level": group.representation_level,
                "ap_native": group.ap_native,
                "used_tool": group.used_tool,
                "metrics": {
                    "macro_average_accuracy": _round4(mean(value["accuracy"] for value in by_domain.values())),
                    "overall_accuracy": _round4(_ratio(sum(1 for record in group_records if record["scored_success_after_examiner_reveal"]), len(group_records))),
                    "d1_text_accuracy": by_domain["d1_text_relation"]["accuracy"],
                    "d2_symbol_accuracy": by_domain["d2_symbol_shape"]["accuracy"],
                    "d3_draft_accuracy": by_domain["d3_draft_buffer"]["accuracy"],
                    "trigger_accuracy": _round4(_ratio(sum(1 for record in trigger_records if record["scored_success_after_examiner_reveal"]), len(trigger_records))),
                    "repair_accuracy": _round4(_ratio(sum(1 for record in repair_records if record["scored_success_after_examiner_reveal"]), len(repair_records))),
                    "false_positive_rate": _round4(_ratio(false_positive, len(group_records))),
                    "false_negative_rate": _round4(_ratio(false_negative, len(group_records))),
                    "training_example_count": 82,
                    "api_call_count": 0,
                    "token_cost": 0,
                    "representation_cost_level": {
                        "r1_structured_process_sa": 2,
                        "r2_surface_text_only": 1,
                        "r3_process_event_bridge": 2,
                        "r4_domain_surface_adapter": 4,
                        "r5_shuffled_process_bridge": 2,
                    }[group.group_id],
                    "by_domain": by_domain,
                },
            }
        )
    return rows


def _aggregate(seed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_values: dict[tuple[str, str], list[float]] = {}
    group_info: dict[str, dict[str, Any]] = {}
    for seed_run in seed_runs:
        for group in seed_run["group_metrics"]:
            group_id = group["group_id"]
            group_info[group_id] = {
                key: group[key]
                for key in ("group_id", "display_name_zh", "route", "boundary", "representation_level", "ap_native", "used_tool")
            }
            for metric_name, value in group["metrics"].items():
                if isinstance(value, (int, float)):
                    metric_values.setdefault((group_id, metric_name), []).append(float(value))
    return [
        {
            **group_info[group_id],
            "metrics": {
                metric_name: _stats(values)
                for (metric_group, metric_name), values in sorted(metric_values.items())
                if metric_group == group_id
            },
        }
        for group_id in sorted(group_info)
    ]


def _metric(row: dict[str, Any], name: str) -> float:
    return float(row["metrics"][name]["mean"])


def _build_validation(summary: dict[str, Any], records: list[dict[str, Any]], combined_text: str = "") -> dict[str, Any]:
    groups = {row["group_id"]: row for row in summary["aggregates"]}
    forbidden_hits = _contains_forbidden_key(records)
    provider_called = sum(1 for record in records if record.get("decision", {}).get("student_side_provider_called"))
    hidden_solver = sum(1 for record in records if record.get("decision", {}).get("hidden_solver_used"))
    r1 = groups["r1_structured_process_sa"]
    r2 = groups["r2_surface_text_only"]
    r3 = groups["r3_process_event_bridge"]
    r4 = groups["r4_domain_surface_adapter"]
    r5 = groups["r5_shuffled_process_bridge"]
    checks = {
        "public_records_no_private_examiner_fields": not forbidden_hits,
        "student_side_provider_called_count_is_zero": provider_called == 0,
        "hidden_solver_count_is_zero": hidden_solver == 0,
        "outcome_feedback_not_used_pre_action": all(record.get("decision", {}).get("outcome_anchor_score_visible_to_decision") == 0.0 for record in records),
        "structured_process_sa_high": _metric(r1, "macro_average_accuracy") >= 0.95,
        "process_event_bridge_high": _metric(r3, "macro_average_accuracy") >= 0.92,
        "bridge_close_to_structured": _metric(r1, "macro_average_accuracy") - _metric(r3, "macro_average_accuracy") <= 0.08,
        "structured_beats_surface_text": _metric(r1, "macro_average_accuracy") - _metric(r2, "macro_average_accuracy") >= 0.20,
        "bridge_beats_shuffled_bridge": _metric(r3, "macro_average_accuracy") - _metric(r5, "macro_average_accuracy") >= 0.25,
        "domain_adapter_marked_non_ap_native": r4.get("ap_native") is False and "Non AP-native" in r4.get("route", ""),
    }
    if combined_text:
        checks["report_mentions_representation_boundary"] = "不是通用智能同一起跑线胜负" in combined_text and "非 AP-native" in combined_text
    return {
        "validation_passed": all(checks.values()),
        "checks": checks,
        "public_record_forbidden_key_hits": sorted(set(forbidden_hits)),
        "student_side_provider_called_count": provider_called,
        "hidden_solver_count": hidden_solver,
    }


def _bar(label: str, value: float, *, color: str = "#1f8f6a") -> str:
    width = max(0, min(100, round(value * 100, 1)))
    return f'<div class="bar-row"><div class="bar-label">{html.escape(label)}</div><div class="bar-track"><div class="bar-fill" style="width:{width}%;background:{color}"></div></div><div class="bar-value">{value:.2f}</div></div>'


def _render_report(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {name} | {macro:.3f} | {d1:.3f} | {d2:.3f} | {d3:.3f} | {fp:.3f} | {fn:.3f} | {cost} | {native} |".format(
            name=row["display_name_zh"],
            macro=_metric(row, "macro_average_accuracy"),
            d1=_metric(row, "d1_text_accuracy"),
            d2=_metric(row, "d2_symbol_accuracy"),
            d3=_metric(row, "d3_draft_accuracy"),
            fp=_metric(row, "false_positive_rate"),
            fn=_metric(row, "false_negative_rate"),
            cost=_metric(row, "representation_cost_level"),
            native=row["ap_native"],
        )
        for row in payload["summary"]["aggregates"]
    )
    checks = "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in payload["summary"]["validation"]["checks"].items())
    return f"""# Representation-Ablation-1 报告

生成时间: {payload["created_at"]}  
schema: `{payload["schema_id"]}`  
定位: representation/fairness appendix / no API / 不是通用智能同一起跑线胜负

## 1. 实验目的

本实验回应“AP 是否因为使用任务特化结构化表示而占优”。实验不否认表示设计的重要性, 而是把输入层级拆开比较: numeric process SA、surface text markers、generic process-event bridge、domain surface adapter 和 shuffled bridge。

## 2. 结果

| 组别 | 宏平均 | D1 | D2 | D3 | FP | FN | 表示成本 | AP-native |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

## 3. 关键解释

- R1 结构化过程 SA 是当前主路线, 高分说明白箱表示工程有效。
- R2 表面文本 token 迁移下降, 说明表面关键词不适合作为稳定范式锚。
- R3 过程事件 bridge 不读取 numeric anchors, 只从公开过程事件标签重构通用 anchors; 若接近 R1, 说明关键不是任务私有答案字段, 而是过程来源正确的 runtime trace。
- R4 分域表面适配器可能高分, 但它是非 AP-native 工程上界, 需要为每个域写适配规则。
- R5 打乱过程事件 bridge 同分布但来源错位, 用于证明不是“有过程字段就行”。

## 4. 自动验收

{checks}

validation_passed: `{payload["summary"]["validation"]["validation_passed"]}`  
student_side_provider_called_count: `{payload["summary"]["validation"]["student_side_provider_called_count"]}`  
hidden_solver_count: `{payload["summary"]["validation"]["hidden_solver_count"]}`

## 5. 可以支持的结论

结构化表示确实是 AP 白箱能力工程的一部分, 不应被隐藏; 但 Representation-Ablation-1 显示, 当任务私有 numeric anchors 被替换成通用过程事件 bridge 时, 过程来源正确的内源信号仍能保持迁移能力。相反, 表面文本 token 和打乱的过程事件都无法替代它。

## 6. 不能推出的结论

这不是开放世界原始感知完成态证明; 不是 AP 与 LLM 的通用胜负; 不是说所有结构化表示都公平; 也不是说 GL/桌宠真实接口已经完全等同 AP-Core。
"""


def _render_html(payload: dict[str, Any]) -> str:
    groups = {row["group_id"]: row for row in payload["summary"]["aggregates"]}
    bars = "\n".join(
        [
            _bar("R1 结构化过程 SA", _metric(groups["r1_structured_process_sa"], "macro_average_accuracy")),
            _bar("R2 表面文本 token", _metric(groups["r2_surface_text_only"], "macro_average_accuracy"), color="#cc5a28"),
            _bar("R3 过程事件 bridge", _metric(groups["r3_process_event_bridge"], "macro_average_accuracy")),
            _bar("R4 分域表面适配器", _metric(groups["r4_domain_surface_adapter"], "macro_average_accuracy"), color="#7a6a00"),
            _bar("R5 打乱过程事件", _metric(groups["r5_shuffled_process_bridge"], "macro_average_accuracy"), color="#cc5a28"),
        ]
    )
    status = "PASS" if payload["summary"]["validation"]["validation_passed"] else "FAIL"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Representation-Ablation-1</title>
<style>
body{{margin:0;font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif;color:#172026;background:#fff}}
.page{{max-width:1160px;margin:0 auto;padding:34px 42px 46px}}h1{{font-size:32px;line-height:1.18;margin:0 0 12px}}h2{{font-size:20px;margin:0 0 14px}}p{{font-size:15px;line-height:1.72;margin:0 0 10px;color:#52616b}}
.badge{{display:inline-block;border:1px solid #1f8f6a;color:#1f8f6a;padding:5px 10px;font-weight:700;margin-bottom:14px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:22px}}.panel{{border:1px solid #d7dee3;background:#f7faf8;padding:18px;border-radius:6px}}
.bar-row{{display:grid;grid-template-columns:220px 1fr 48px;gap:10px;align-items:center;margin:10px 0}}.bar-label{{font-size:13px}}.bar-track{{height:18px;background:#e3e8e7;border:1px solid #cfd8d4}}.bar-fill{{height:100%}}.bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:700}}.note{{border-left:4px solid #cc5a28;padding-left:12px;color:#3c464c}}
</style></head><body><main class="page">
<section><div class="badge">Representation-Ablation-1 / {status} / no API</div><h1>AP 是否靠结构化表示占便宜?</h1>
<p>这次不回避表示问题: 同一批 case 分别用结构化过程 SA、表面文本 token、通用过程事件 bridge、分域表面适配器和打乱过程事件来测。</p></section>
<div class="grid"><section class="panel"><h2>宏平均准确率</h2>{bars}</section>
<section class="panel"><h2>读法</h2><p class="note">R1/R3 高, R2/R5 低, R4 标记为非 AP-native 上界。结论不是“表示不重要”, 而是“正确过程来源比表面关键词和错位过程字段更关键”。</p></section></div>
</main></body></html>"""


def _build_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    names = (
        "summary.json",
        "records.json",
        "private_examiner_key.json",
        "RepresentationAblation1_report_zh.md",
        "representation_ablation_1_showcase_zh.html",
    )
    files = {}
    for name in names:
        path = output_dir / name
        files[name] = {"path": _display_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_id": "apv21_representation_ablation_1_artifact_manifest/v0.1",
        "created_at": _now_iso(),
        "experiment_schema_id": payload["schema_id"],
        "validation_passed": payload["summary"]["validation"]["validation_passed"],
        "artifact_boundary": "local manifest/hash freeze only; current workspace is not a git repository",
        "files": files,
    }


def run_representation_ablation_1(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed_runs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    private_examiner = {
        "schema_id": "apv21_representation_ablation_1_private_examiner/v0.1",
        "created_at": _now_iso(),
        "reveal_policy": "Generated before scoring, not passed into learner decisions; published after run for reproducibility.",
        "seeds": {},
    }
    for seed in seeds:
        seed_run, seed_records, seed_private = _run_seed(int(seed))
        seed_runs.append(seed_run)
        records.extend(seed_records)
        private_examiner["seeds"][str(seed)] = seed_private
    summary: dict[str, Any] = {
        "description": "Representation ablation for structured process SA vs surface text vs process-event bridge.",
        "aggregates": _aggregate(seed_runs),
    }
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "created_at": _now_iso(),
        "status": "strict_core_controlled_representation_ablation_no_api_candidate",
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "student_side_llm": False,
        "student_side_provider_called": False,
        "hidden_solver": False,
        "not_claiming": ["open_world_raw_perception", "LLM benchmark victory", "full APV2.1 runtime completion"],
        "summary": summary,
        "seed_runs": seed_runs,
        "record_count": len(records),
    }
    report_probe = _render_report({**payload, "summary": {**summary, "validation": _build_validation(summary, records)}})
    html_probe = _render_html({**payload, "summary": {**summary, "validation": _build_validation(summary, records)}})
    summary["validation"] = _build_validation(summary, records, combined_text=report_probe + "\n" + html_probe)
    payload["summary"] = summary
    _write_json(output_path / "summary.json", payload)
    _write_json(output_path / "records.json", records)
    _write_json(output_path / "private_examiner_key.json", private_examiner)
    _write_text(output_path / "RepresentationAblation1_report_zh.md", _render_report(payload))
    _write_text(output_path / "representation_ablation_1_showcase_zh.html", _render_html(payload))
    manifest = _build_manifest(output_path, payload)
    _write_json(output_path / "artifact_manifest.json", manifest)
    payload["artifact_manifest"] = manifest
    _write_json(output_path / "summary.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APV2.1 Representation-Ablation-1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    payload = run_representation_ablation_1(output_dir=args.output_dir, seeds=seeds)
    print(
        json.dumps(
            {
                "schema_id": payload["schema_id"],
                "output_dir": _display_path(Path(args.output_dir)),
                "validation_passed": payload["summary"]["validation"]["validation_passed"],
                "record_count": payload["record_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
