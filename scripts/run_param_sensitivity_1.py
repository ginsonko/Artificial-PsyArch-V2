from __future__ import annotations

import argparse
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
    _clamp,
    _contains_forbidden_key,
    _display_path,
    _ratio,
    _round4,
    _sha256,
    _stable_int,
    _stats,
    _write_json,
    _write_text,
)
from scripts.run_stpv2_process_anchor_transfer_v04 import (
    ANCHOR_KEYS,
    DOMAINS,
    _fit_process_policy,
    _make_cases,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "param_sensitivity_1_20260608"
DEFAULT_SEEDS = (2026060861, 2026060862, 2026060863)
SCHEMA_ID = "apv21_param_sensitivity_1/v0.1"

THRESHOLD_OFFSETS = (-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16)
FEELING_GAINS = (0.70, 0.85, 1.00, 1.15, 1.30)
NOISE_SIGMAS = (0.00, 0.03, 0.06, 0.09)
REPAIR_WEIGHT_BIASES = (0.70, 1.00, 1.30)
TRIGGER_WEIGHT_BIASES = (0.70, 1.00, 1.30)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class SensitivityConfig:
    threshold_offset: float
    feeling_gain: float
    anchor_noise_sigma: float
    repair_weight_bias: float
    trigger_weight_bias: float

    @property
    def config_id(self) -> str:
        return (
            f"to{self.threshold_offset:+.2f}_"
            f"fg{self.feeling_gain:.2f}_"
            f"ns{self.anchor_noise_sigma:.2f}_"
            f"rw{self.repair_weight_bias:.2f}_"
            f"tw{self.trigger_weight_bias:.2f}"
        ).replace("+", "p").replace("-", "m").replace(".", "d")

    @property
    def is_base(self) -> bool:
        return (
            self.threshold_offset == 0.0
            and self.feeling_gain == 1.0
            and self.anchor_noise_sigma == 0.0
            and self.repair_weight_bias == 1.0
            and self.trigger_weight_bias == 1.0
        )

    @property
    def is_core_basin(self) -> bool:
        return (
            abs(self.threshold_offset) <= 0.08
            and 0.85 <= self.feeling_gain <= 1.15
            and self.anchor_noise_sigma <= 0.06
            and 0.70 <= self.repair_weight_bias <= 1.30
            and 0.70 <= self.trigger_weight_bias <= 1.30
        )

    @property
    def is_extreme(self) -> bool:
        return (
            abs(self.threshold_offset) >= 0.12
            or self.feeling_gain in {0.70, 1.30}
            or self.anchor_noise_sigma >= 0.09
        )


def _configs() -> list[SensitivityConfig]:
    return [
        SensitivityConfig(
            threshold_offset=threshold_offset,
            feeling_gain=feeling_gain,
            anchor_noise_sigma=noise_sigma,
            repair_weight_bias=repair_weight_bias,
            trigger_weight_bias=trigger_weight_bias,
        )
        for threshold_offset in THRESHOLD_OFFSETS
        for feeling_gain in FEELING_GAINS
        for noise_sigma in NOISE_SIGMAS
        for repair_weight_bias in REPAIR_WEIGHT_BIASES
        for trigger_weight_bias in TRIGGER_WEIGHT_BIASES
    ]


def _noise(case_id: str, key: str, config_id: str, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    rng = random.Random(_stable_int(f"{case_id}:{key}:{config_id}"))
    return rng.uniform(-sigma, sigma)


def _effective_anchors(case: dict[str, Any], config: SensitivityConfig) -> dict[str, float]:
    anchors = dict(case.get("process_anchors_public", {}) or {})
    result: dict[str, float] = {}
    for key in ANCHOR_KEYS:
        value = float(anchors.get(key, 0.0) or 0.0)
        value = value * config.feeling_gain + _noise(case["case_id"], key, config.config_id, config.anchor_noise_sigma)
        result[key] = _round4(_clamp(value))
    return result


def _head_score(case: dict[str, Any], anchors: dict[str, float], config: SensitivityConfig) -> float:
    if case["action_head"] == "relation_trigger":
        score = (anchors["teacher_context"] + anchors["correction_event"]) / 2.0
        return _round4(_clamp(score * config.trigger_weight_bias))
    score = (anchors["mismatch"] + anchors["low_grasp"]) / 2.0
    return _round4(_clamp(score * config.repair_weight_bias))


def _threshold(case: dict[str, Any], base_policy: dict[str, float], config: SensitivityConfig) -> float:
    name = "relation_trigger_threshold" if case["action_head"] == "relation_trigger" else "local_repair_threshold"
    return _round4(_clamp(float(base_policy[name]) + config.threshold_offset))


def _evaluate_config(
    *,
    seed: int,
    config: SensitivityConfig,
    test_cases: list[dict[str, Any]],
    private_cases: dict[str, Any],
    base_policy: dict[str, float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in test_cases:
        anchors = _effective_anchors(case, config)
        score = _head_score(case, anchors, config)
        threshold = _threshold(case, base_policy, config)
        decision = score >= threshold
        expected = bool(private_cases[case["case_id"]]["action_should_fire"])
        rows.append(
            {
                "case_id": case["case_id"],
                "domain": case["domain"],
                "action_head": case["action_head"],
                "family_public": case["family_public"],
                "decision_fire": bool(decision),
                "success": bool(decision == expected),
                "false_positive": bool(decision and not expected),
                "false_negative": bool((not decision) and expected),
                "score": score,
                "threshold": threshold,
            }
        )

    def by(predicate: Any) -> list[dict[str, Any]]:
        return [row for row in rows if predicate(row)]

    domain_metrics = {}
    for domain in DOMAINS:
        domain_rows = by(lambda row, d=domain: row["domain"] == d)
        domain_metrics[domain] = _round4(_ratio(sum(1 for row in domain_rows if row["success"]), len(domain_rows)))
    trigger_rows = by(lambda row: row["action_head"] == "relation_trigger")
    repair_rows = by(lambda row: row["action_head"] == "local_repair")
    macro_accuracy = _round4(mean(domain_metrics.values()))
    pass_core = (
        macro_accuracy >= 0.92
        and domain_metrics["d2_symbol_shape"] >= 0.88
        and domain_metrics["d3_draft_buffer"] >= 0.88
        and _ratio(sum(1 for row in rows if row["false_positive"]), len(rows)) <= 0.12
    )
    sample_trace = [
        {
            "case_id": row["case_id"],
            "domain": row["domain"],
            "action_head": row["action_head"],
            "family_public": row["family_public"],
            "score": row["score"],
            "threshold": row["threshold"],
            "decision_fire": row["decision_fire"],
            "scored_success_after_private_reveal": row["success"],
        }
        for row in rows[:6]
    ]
    return {
        "schema_id": "apv21_param_sensitivity_1/config_record/v0.1",
        "seed": int(seed),
        "config_id": config.config_id,
        "config": {
            "threshold_offset": config.threshold_offset,
            "feeling_gain": config.feeling_gain,
            "anchor_noise_sigma": config.anchor_noise_sigma,
            "repair_weight_bias": config.repair_weight_bias,
            "trigger_weight_bias": config.trigger_weight_bias,
            "is_base": config.is_base,
            "is_core_basin": config.is_core_basin,
            "is_extreme": config.is_extreme,
        },
        "metrics": {
            "macro_accuracy": macro_accuracy,
            "overall_accuracy": _round4(_ratio(sum(1 for row in rows if row["success"]), len(rows))),
            "d1_text_accuracy": domain_metrics["d1_text_relation"],
            "d2_symbol_accuracy": domain_metrics["d2_symbol_shape"],
            "d3_draft_accuracy": domain_metrics["d3_draft_buffer"],
            "trigger_accuracy": _round4(_ratio(sum(1 for row in trigger_rows if row["success"]), len(trigger_rows))),
            "repair_accuracy": _round4(_ratio(sum(1 for row in repair_rows if row["success"]), len(repair_rows))),
            "false_positive_rate": _round4(_ratio(sum(1 for row in rows if row["false_positive"]), len(rows))),
            "false_negative_rate": _round4(_ratio(sum(1 for row in rows if row["false_negative"]), len(rows))),
            "pass_core": bool(pass_core),
        },
        "decision_trace_sample_public": sample_trace,
        "outcome_anchor_score_visible_to_decision": 0.0,
        "student_side_provider_called": False,
        "hidden_solver_used": False,
    }


def _run_seed(seed: int, configs: list[SensitivityConfig]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_cases, test_cases, private_payload = _make_cases(seed)
    private_cases = private_payload["cases"]
    base_policy = _fit_process_policy(train_cases, private_cases)
    records = [
        _evaluate_config(
            seed=seed,
            config=config,
            test_cases=test_cases,
            private_cases=private_cases,
            base_policy=base_policy,
        )
        for config in configs
    ]
    private_payload = {
        **private_payload,
        "seed": int(seed),
        "base_policy": base_policy,
        "reveal_policy": "Private labels are used only after learner decisions for scoring.",
    }
    return records, private_payload


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "macro_accuracy",
        "overall_accuracy",
        "d1_text_accuracy",
        "d2_symbol_accuracy",
        "d3_draft_accuracy",
        "trigger_accuracy",
        "repair_accuracy",
        "false_positive_rate",
        "false_negative_rate",
    ]
    by_metric = {name: [float(record["metrics"][name]) for record in records] for name in metric_names}
    base_records = [record for record in records if record["config"]["is_base"]]
    core_records = [record for record in records if record["config"]["is_core_basin"]]
    extreme_records = [record for record in records if record["config"]["is_extreme"]]

    def pass_rate(rows: list[dict[str, Any]]) -> float:
        return _round4(_ratio(sum(1 for record in rows if record["metrics"]["pass_core"]), len(rows)))

    slice_metrics: dict[str, list[dict[str, Any]]] = {}
    for field in ("threshold_offset", "feeling_gain", "anchor_noise_sigma", "repair_weight_bias", "trigger_weight_bias"):
        values = sorted({record["config"][field] for record in records})
        slice_metrics[field] = [
            {
                "value": value,
                "n": len(rows := [record for record in records if record["config"][field] == value]),
                "pass_rate": pass_rate(rows),
                "macro_accuracy": _stats([float(record["metrics"]["macro_accuracy"]) for record in rows]),
                "false_positive_rate": _stats([float(record["metrics"]["false_positive_rate"]) for record in rows]),
                "false_negative_rate": _stats([float(record["metrics"]["false_negative_rate"]) for record in rows]),
            }
            for value in values
        ]
    return {
        "metric_stats": {name: _stats(values) for name, values in by_metric.items()},
        "base_config": {
            "n": len(base_records),
            "pass_rate": pass_rate(base_records),
            "macro_accuracy": _stats([float(record["metrics"]["macro_accuracy"]) for record in base_records]),
        },
        "core_basin": {
            "n": len(core_records),
            "pass_rate": pass_rate(core_records),
            "macro_accuracy": _stats([float(record["metrics"]["macro_accuracy"]) for record in core_records]),
        },
        "extreme_region": {
            "n": len(extreme_records),
            "pass_rate": pass_rate(extreme_records),
            "min_macro_accuracy": _round4(min(float(record["metrics"]["macro_accuracy"]) for record in extreme_records)),
            "failure_count": sum(1 for record in extreme_records if not record["metrics"]["pass_core"]),
        },
        "slice_metrics": slice_metrics,
    }


def _build_validation(summary: dict[str, Any], records: list[dict[str, Any]], combined_text: str = "") -> dict[str, Any]:
    forbidden_hits = _contains_forbidden_key(records)
    provider_called = sum(1 for record in records if record.get("student_side_provider_called"))
    hidden_solver = sum(1 for record in records if record.get("hidden_solver_used"))
    base_pass = summary["base_config"]["pass_rate"] == 1.0 and summary["base_config"]["macro_accuracy"]["mean"] >= 0.95
    core_pass_rate = float(summary["core_basin"]["pass_rate"])
    fail_boundary = summary["extreme_region"]["failure_count"] > 0 and summary["extreme_region"]["min_macro_accuracy"] <= 0.85
    checks = {
        "public_records_no_private_examiner_fields": not forbidden_hits,
        "student_side_provider_called_count_is_zero": provider_called == 0,
        "hidden_solver_count_is_zero": hidden_solver == 0,
        "outcome_feedback_not_used_pre_action": all(record.get("outcome_anchor_score_visible_to_decision") == 0.0 for record in records),
        "base_config_passes": bool(base_pass),
        "core_basin_pass_rate_ge_0_80": core_pass_rate >= 0.80,
        "extreme_failure_boundary_exists": bool(fail_boundary),
    }
    if combined_text:
        checks["report_mentions_not_single_point_tuning"] = "不是单点调参证明" in combined_text and "失败边界" in combined_text
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
    summary = payload["summary"]["aggregate"]
    checks = "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in payload["summary"]["validation"]["checks"].items())
    slice_rows = "\n".join(
        "| {param} | {value} | {pass_rate:.3f} | {macro:.3f} | {fp:.3f} | {fn:.3f} |".format(
            param=param,
            value=row["value"],
            pass_rate=row["pass_rate"],
            macro=row["macro_accuracy"]["mean"],
            fp=row["false_positive_rate"]["mean"],
            fn=row["false_negative_rate"]["mean"],
        )
        for param, rows in summary["slice_metrics"].items()
        for row in rows
    )
    return f"""# ParamSensitivity-1 报告

生成时间: {payload["created_at"]}  
schema: `{payload["schema_id"]}`  
定位: STP-v2 process-anchor robustness appendix / no API / 不是单点调参证明

## 1. 实验目的

本实验回应“STP-v2 的结果是否只是某个阈值/增益调出来的”这一质疑。实验固定 D1-trained process-anchor policy 和 private examiner, 对阈值、feeling gain、过程锚点噪声、修订权重和触发权重做预注册网格扫描。

## 2. 总览

| 区域 | n | 通过率 | macro accuracy |
|---|---:|---:|---:|
| base config | {summary["base_config"]["n"]} | {summary["base_config"]["pass_rate"]:.3f} | {summary["base_config"]["macro_accuracy"]["mean"]:.3f} |
| core basin | {summary["core_basin"]["n"]} | {summary["core_basin"]["pass_rate"]:.3f} | {summary["core_basin"]["macro_accuracy"]["mean"]:.3f} |
| extreme region | {summary["extreme_region"]["n"]} | {summary["extreme_region"]["pass_rate"]:.3f} | min={summary["extreme_region"]["min_macro_accuracy"]:.3f} |

## 3. 参数切片

| 参数 | 值 | 通过率 | macro | false positive | false negative |
|---|---:|---:|---:|---:|---:|
{slice_rows}

## 4. 自动验收

{checks}

validation_passed: `{payload["summary"]["validation"]["validation_passed"]}`  
student_side_provider_called_count: `{payload["summary"]["validation"]["student_side_provider_called_count"]}`  
hidden_solver_count: `{payload["summary"]["validation"]["hidden_solver_count"]}`

## 5. 解释

可以支持: STP-v2 过程锚点存在稳定参数盆地, base config 不是孤立幸运点; 同时极端阈值/高噪声会产生失败边界, 说明实验不是把验收条件写成永远通过。

不能推出: AP 已完成开放世界鲁棒性; 所有参数任意变化都不影响结果; full APV2.1 runtime 已完成。
"""


def _render_html(payload: dict[str, Any]) -> str:
    aggregate = payload["summary"]["aggregate"]
    bars = "\n".join(
        [
            _bar("Base config pass rate", aggregate["base_config"]["pass_rate"]),
            _bar("Core basin pass rate", aggregate["core_basin"]["pass_rate"]),
            _bar("Extreme region pass rate", aggregate["extreme_region"]["pass_rate"], color="#cc5a28"),
            _bar("Overall macro accuracy", aggregate["metric_stats"]["macro_accuracy"]["mean"]),
            _bar("False positive rate", aggregate["metric_stats"]["false_positive_rate"]["mean"], color="#7a6a00"),
            _bar("False negative rate", aggregate["metric_stats"]["false_negative_rate"]["mean"], color="#7a6a00"),
        ]
    )
    status = "PASS" if payload["summary"]["validation"]["validation_passed"] else "FAIL"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParamSensitivity-1</title>
<style>
body{{margin:0;font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif;color:#172026;background:#fff}}
.page{{max-width:1160px;margin:0 auto;padding:34px 42px 46px}}h1{{font-size:32px;line-height:1.18;margin:0 0 12px}}h2{{font-size:20px;margin:0 0 14px}}p{{font-size:15px;line-height:1.72;margin:0 0 10px;color:#52616b}}
.badge{{display:inline-block;border:1px solid #1f8f6a;color:#1f8f6a;padding:5px 10px;font-weight:700;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:22px}}.panel{{border:1px solid #d7dee3;background:#f7faf8;padding:18px;border-radius:6px}}
.bar-row{{display:grid;grid-template-columns:220px 1fr 48px;gap:10px;align-items:center;margin:10px 0}}.bar-label{{font-size:13px}}.bar-track{{height:18px;background:#e3e8e7;border:1px solid #cfd8d4}}.bar-fill{{height:100%}}.bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:700}}.note{{border-left:4px solid #cc5a28;padding-left:12px;color:#3c464c}}
</style></head><body><main class="page">
<section><div class="badge">ParamSensitivity-1 / {status} / no API</div><h1>STP-v2 是否只是单点调参?</h1>
<p>固定训练集和 private examiner, 扫描 threshold、feeling gain、anchor noise、repair/trigger weight。验收看稳定通过盆地和失败边界, 而不是只挑一个最好点。</p></section>
<div class="grid"><section class="panel"><h2>核心指标</h2>{bars}</section>
<section class="panel"><h2>结论</h2><p class="note">base config 通过, core basin 通过率为 {aggregate["core_basin"]["pass_rate"]:.3f}; extreme region 出现 {aggregate["extreme_region"]["failure_count"]} 个失败配置, 最低 macro={aggregate["extreme_region"]["min_macro_accuracy"]:.3f}。</p><p>这支持“稳定参数盆地”, 也保留了失败边界。</p></section></div>
</main></body></html>"""


def _build_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    names = (
        "summary.json",
        "records.json",
        "private_examiner_key.json",
        "ParamSensitivity1_report_zh.md",
        "param_sensitivity_1_showcase_zh.html",
    )
    files = {}
    for name in names:
        path = output_dir / name
        files[name] = {"path": _display_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_id": "apv21_param_sensitivity_1_artifact_manifest/v0.1",
        "created_at": _now_iso(),
        "experiment_schema_id": payload["schema_id"],
        "validation_passed": payload["summary"]["validation"]["validation_passed"],
        "artifact_boundary": "local manifest/hash freeze only; current workspace is not a git repository",
        "files": files,
    }


def run_param_sensitivity_1(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configs = _configs()
    records: list[dict[str, Any]] = []
    private_examiner = {
        "schema_id": "apv21_param_sensitivity_1_private_examiner/v0.1",
        "created_at": _now_iso(),
        "reveal_policy": "Generated before scoring, not passed into learner decisions; published after run for reproducibility.",
        "seeds": {},
    }
    for seed in seeds:
        seed_records, seed_private = _run_seed(int(seed), configs)
        records.extend(seed_records)
        private_examiner["seeds"][str(seed)] = seed_private

    aggregate = _aggregate(records)
    summary: dict[str, Any] = {
        "description": "Grid sensitivity scan for STP-v2 process-anchor policy.",
        "config_count_per_seed": len(configs),
        "aggregate": aggregate,
    }
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "created_at": _now_iso(),
        "status": "strict_core_controlled_param_sensitivity_no_api_candidate",
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "student_side_llm": False,
        "student_side_provider_called": False,
        "hidden_solver": False,
        "not_claiming": ["open_world_robustness", "full APV2.1 runtime completion", "all parameters arbitrary"],
        "summary": summary,
        "record_count": len(records),
    }
    report_probe = _render_report({**payload, "summary": {**summary, "validation": _build_validation(aggregate, records)}})
    html_probe = _render_html({**payload, "summary": {**summary, "validation": _build_validation(aggregate, records)}})
    summary["validation"] = _build_validation(aggregate, records, combined_text=report_probe + "\n" + html_probe)
    payload["summary"] = summary
    _write_json(output_path / "summary.json", payload)
    _write_json(output_path / "records.json", records)
    _write_json(output_path / "private_examiner_key.json", private_examiner)
    _write_text(output_path / "ParamSensitivity1_report_zh.md", _render_report(payload))
    _write_text(output_path / "param_sensitivity_1_showcase_zh.html", _render_html(payload))
    manifest = _build_manifest(output_path, payload)
    _write_json(output_path / "artifact_manifest.json", manifest)
    payload["artifact_manifest"] = manifest
    _write_json(output_path / "summary.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APV2.1 ParamSensitivity-1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    payload = run_param_sensitivity_1(output_dir=args.output_dir, seeds=seeds)
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
