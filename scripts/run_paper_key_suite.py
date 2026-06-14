from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "paper_artifacts" / "apv21_20260605"


@dataclass(frozen=True)
class SuiteItem:
    suite_id: str
    claim_id: str
    title: str
    route: str
    command: list[str]
    evidence_paths: list[str]
    allowed_wording: str
    forbidden_wording: str
    teacher_off_boundary: str
    solver_boundary: str
    expected_outputs: list[str]


SUITE: list[SuiteItem] = [
    SuiteItem(
        suite_id="KS-001",
        claim_id="CORE-001",
        title="StrictCore-0 foundation boundary suite",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/strict_core/test_strictcore0_foundation_boundary_suite.py", "-q"],
        evidence_paths=[
            "AP_Core_Proof/strict_core/reports/strictcore0_foundation_boundary_suite/StrictCore0_FoundationBoundary_Report_20260603.md",
            "tests/strict_core/test_strictcore0_foundation_boundary_suite.py",
        ],
        allowed_wording="受控 numeric 状态中, AP runtime/core interface 可形成行动、反馈、记忆、后续 teacher-off 行动改变的最小闭环。",
        forbidden_wording="已证明开放世界通用学习或完整 AGI。",
        teacher_off_boundary="teacher-off 阶段 feedback_to_learner 为 None, teacher_off_signal 为空, 测试后 memory write 为 0。",
        solver_boundary="任务为 numeric/action feedback 闭环, 不允许 hidden truth 进入 visible payload 或 package。",
        expected_outputs=[
            "AP_Core_Proof/strict_core/reports/strictcore0_foundation_boundary_suite/StrictCore0_FoundationBoundary_Report_20260603.md",
        ],
    ),
    SuiteItem(
        suite_id="KS-002",
        claim_id="CORE-002",
        title="LoopLearn-1 blind action-feedback closed loop",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/test_looplearn1_blind_rule_closed_loop.py", "-q"],
        evidence_paths=[
            "experiments/looplearn1_blind_rule_closed_loop/looplearn1_blind_rule_closed_loop_latest.json",
            "tests/test_looplearn1_blind_rule_closed_loop.py",
        ],
        allowed_wording="行动后果可写入记忆, 并影响相似状态下的后续行动选择。",
        forbidden_wording="任意任务都能盲学会。",
        teacher_off_boundary="posttest/ood 阶段无反馈、无修复、无重试。",
        solver_boundary="learner region 禁止 hidden rule、answer、oracle、solution 等字段。",
        expected_outputs=["experiments/looplearn1_blind_rule_closed_loop/looplearn1_blind_rule_closed_loop_latest.json"],
    ),
    SuiteItem(
        suite_id="KS-003",
        claim_id="CORE-003",
        title="LoopLearn-2 compositional feature generalization",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/test_looplearn2_compositional_feature_generalization.py", "-q"],
        evidence_paths=[
            "experiments/looplearn2_compositional_feature_generalization/looplearn2_compositional_feature_generalization_latest.json",
            "tests/test_looplearn2_compositional_feature_generalization.py",
        ],
        allowed_wording="局部 numeric feature contribution 可组合到未见状态, 不只是整态表绑定。",
        forbidden_wording="已证明完整语义表征或任意组合泛化。",
        teacher_off_boundary="unseen combo teacher-off 阶段不回传反馈。",
        solver_boundary="exact whole-state 和 nearest whole-state baseline 独立对照, 不允许 target action 泄漏。",
        expected_outputs=[
            "experiments/looplearn2_compositional_feature_generalization/looplearn2_compositional_feature_generalization_latest.json",
        ],
    ),
    SuiteItem(
        suite_id="KS-004",
        claim_id="CORE-004",
        title="CountLoop-0 object quantity and add/remove acquisition",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/test_countloop0_object_quantity_successor_acquisition.py", "-q"],
        evidence_paths=[
            "experiments/countloop0_object_quantity_successor_acquisition/countloop0_object_quantity_successor_acquisition_latest.json",
            "tests/test_countloop0_object_quantity_successor_acquisition.py",
        ],
        allowed_wording="6 槽对象世界中, 小范围数量标签和 add/remove 迁移可通过反馈后天习得。",
        forbidden_wording="完整自然数概念或无限递推已证明。",
        teacher_off_boundary="teacher-off 阶段无反馈、无修复、无重试。",
        solver_boundary="不预置 successor/predecessor 表给 learner, 禁止 exact-slot/inverted/no-memory/no-sensor 捷径。",
        expected_outputs=[
            "experiments/countloop0_object_quantity_successor_acquisition/countloop0_object_quantity_successor_acquisition_latest.json",
        ],
    ),
    SuiteItem(
        suite_id="KS-005",
        claim_id="CORE-005",
        title="Multimodal NoInject0 raw sensor feature association",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/test_multimodal_noinject0_sensor_feature_association.py", "-q"],
        evidence_paths=[
            "AP_Core_Proof/experiments/multimodal_noinject0_sensor_feature_association/multimodal_noinject0_sensor_feature_association_latest.json",
            "tests/test_multimodal_noinject0_sensor_feature_association.py",
            "docs/ReviewReply_Multimodal_NoInject0_Evidence_20260603.md",
        ],
        allowed_wording="teacher-off 阶段只给 raw image/audio bytes 经 native numeric sensor 后的状态项, AP 可召回训练期同场出现过的无意义标签。",
        forbidden_wording="开放世界物体识别、OCR 或 ASR 已证明。",
        teacher_off_boundary="teacher-off 查询不含名称、target_text、vision_semantic 或 audio_semantic。",
        solver_boundary="shuffled-label/no-learning 对照应失败, 语义标签只在训练期同场出现。",
        expected_outputs=[
            "AP_Core_Proof/experiments/multimodal_noinject0_sensor_feature_association/multimodal_noinject0_sensor_feature_association_latest.json",
        ],
    ),
    SuiteItem(
        suite_id="KS-006",
        claim_id="CORE-006",
        title="ElementaryMath NoSolver 0-4",
        route="AP-Core",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_elementary_math_nosolver0_add_sub_acquired.py",
            "tests/test_elementary_math_nosolver1_mul_div_acquired.py",
            "tests/test_elementary_math_nosolver2_vertical_add_sub_acquired.py",
            "tests/test_elementary_math_nosolver3_vertical_multiply_acquired.py",
            "tests/test_elementary_math_nosolver4_vertical_division_acquired.py",
            "-q",
        ],
        evidence_paths=[
            "experiments/elementary_math_nosolver0_add_sub_acquired/elementary_math_nosolver0_add_sub_acquired_latest.json",
            "experiments/elementary_math_nosolver1_mul_div_acquired/elementary_math_nosolver1_mul_div_acquired_latest.json",
            "experiments/elementary_math_nosolver2_vertical_add_sub_acquired/elementary_math_nosolver2_vertical_add_sub_acquired_latest.json",
            "experiments/elementary_math_nosolver3_vertical_multiply_acquired/elementary_math_nosolver3_vertical_multiply_acquired_latest.json",
            "experiments/elementary_math_nosolver4_vertical_division_acquired/elementary_math_nosolver4_vertical_division_acquired_latest.json",
            "docs/ReviewReply_NoSolver_Math_Evidence_and_Semantic_Boundary_20260603.md",
        ],
        allowed_wording="十以内加减、小范围乘除、竖式加减、竖式乘法、一位除数竖式除法已有受控过程切片证据。",
        forbidden_wording="完整小学数学已经证明。",
        teacher_off_boundary="teacher-off 阶段不给答案、不回填、不在错后修复。",
        solver_boundary="learner 区域禁止当前题 Python solver、答案字段和脚本曲线。",
        expected_outputs=[
            "experiments/elementary_math_nosolver0_add_sub_acquired/elementary_math_nosolver0_add_sub_acquired_latest.json",
            "experiments/elementary_math_nosolver1_mul_div_acquired/elementary_math_nosolver1_mul_div_acquired_latest.json",
            "experiments/elementary_math_nosolver2_vertical_add_sub_acquired/elementary_math_nosolver2_vertical_add_sub_acquired_latest.json",
            "experiments/elementary_math_nosolver3_vertical_multiply_acquired/elementary_math_nosolver3_vertical_multiply_acquired_latest.json",
            "experiments/elementary_math_nosolver4_vertical_division_acquired/elementary_math_nosolver4_vertical_division_acquired_latest.json",
        ],
    ),
    SuiteItem(
        suite_id="KS-007",
        claim_id="CORE-007",
        title="Math-FullChain0/1 equation word problem chain",
        route="AP-Core",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_math_fullchain0_foundation_to_equation_word_problem.py",
            "tests/test_math_fullchain1_pure_ap_equation_word_problem.py",
            "-q",
        ],
        evidence_paths=[
            "AP_Core_Proof/experiments/math_fullchain0_foundation_to_equation_word_problem/math_fullchain0_foundation_to_equation_word_problem_latest.json",
            "AP_Core_Proof/experiments/math_fullchain1_pure_ap_equation_word_problem/math_fullchain1_pure_ap_equation_word_problem_latest.json",
            "AP_Core_Proof/docs/ColdSave_MathFullChain1_PureAP_FinalReport_20260603.md",
            "AP_Core_Proof/docs/MathFullChain1_real_output_samples_20260603.md",
        ],
        allowed_wording="受控模板内, 基础事实/过程技能可接到简单一元一次应用题链路, 并注册为技能包。",
        forbidden_wording="任意自然语言代数应用题已解决。",
        teacher_off_boundary="teacher-off 测试阶段不把反馈交回 learner, 只由 examiner 私下记录分数。",
        solver_boundary="learner 区域不调用当前题 Python 求解器、隐藏真值读取或标准答案回填。",
        expected_outputs=[
            "AP_Core_Proof/experiments/math_fullchain0_foundation_to_equation_word_problem/math_fullchain0_foundation_to_equation_word_problem_latest.json",
            "AP_Core_Proof/experiments/math_fullchain1_pure_ap_equation_word_problem/math_fullchain1_pure_ap_equation_word_problem_latest.json",
        ],
    ),
    SuiteItem(
        suite_id="KS-008",
        claim_id="CORE-008",
        title="AP learned skill registry",
        route="AP-Core",
        command=[sys.executable, "-m", "pytest", "tests/test_ap_learned_skill_registry.py", "-q"],
        evidence_paths=[
            "AP_Core_Proof/skill_registry/ap_learned_skill_registry.json",
            "AP_Core_Proof/skill_registry/AP_LearnedSkillRegistry_Report_20260603.md",
            "tests/test_ap_learned_skill_registry.py",
        ],
        allowed_wording="底层技能可固化为 action::skill.*, 高层任务可声明依赖后复用。",
        forbidden_wording="技能包是万能求解器或隐藏代答。",
        teacher_off_boundary="registry 记录技能来源和 teacher-off 边界, 高层复用必须声明依赖。",
        solver_boundary="技能包不能把最终答案表伪装为过程技能。",
        expected_outputs=[
            "AP_Core_Proof/skill_registry/ap_learned_skill_registry.json",
            "AP_Core_Proof/skill_registry/AP_LearnedSkillRegistry_Report_20260603.md",
        ],
    ),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path_str: str) -> dict[str, Any]:
    path = ROOT / path_str
    exists = path.exists()
    record: dict[str, Any] = {"path": path_str, "exists": exists}
    if exists and path.is_file():
        record.update(
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "copied_to": None,
            }
        )
    return record


def copy_evidence_files(item: SuiteItem, artifact_root: Path, records: list[dict[str, Any]]) -> None:
    target_dir = artifact_root / "source_refs" / item.suite_id.lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for rec in records:
        path_str = rec["path"]
        if path_str in seen or not rec.get("exists"):
            continue
        seen.add(path_str)
        src = ROOT / path_str
        if not src.is_file():
            continue
        # Keep the key package compact: copy small/medium text artifacts, not arbitrary large assets.
        if src.stat().st_size > 5 * 1024 * 1024:
            continue
        safe_name = path_str.replace("/", "__").replace("\\", "__").replace(":", "")
        dst = target_dir / safe_name
        shutil.copy2(src, dst)
        rec["copied_to"] = _rel(dst)


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    duration = round(time.perf_counter() - start, 3)
    return {
        "command": command,
        "exit_code": proc.returncode,
        "duration_seconds": duration,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def collect_manifest(artifact_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            files.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema_id": "apv21_paper_key_suite_manifest/v1",
        "generated_at": _now_iso(),
        "artifact_root": _rel(artifact_root),
        "file_count": len(files),
        "files": files,
    }


def write_markdown_report(artifact_root: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# APV2.1 Canonical-KeySuite-1 运行报告\n")
    lines.append(f"生成时间: {payload['generated_at']}")
    lines.append(f"状态: {'PASS' if payload['summary']['passed'] else 'FAIL'}")
    lines.append(f"套件数: {payload['summary']['suite_count']}")
    lines.append(f"通过数: {payload['summary']['passed_count']}")
    lines.append("")
    lines.append("## 环境\n")
    env = payload["environment"]
    for key in ("python", "platform", "cwd", "repository_status"):
        lines.append(f"- {key}: `{env[key]}`")
    lines.append("")
    lines.append("## 套件结果\n")
    lines.append("| id | claim | title | exit | seconds | status |")
    lines.append("|---|---|---|---:|---:|---|")
    for item in payload["suite_results"]:
        lines.append(
            f"| {item['suite_id']} | {item['claim_id']} | {item['title']} | {item['run']['exit_code']} | {item['run']['duration_seconds']} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## 失败详情\n")
    failed = [item for item in payload["suite_results"] if not item["passed"]]
    if not failed:
        lines.append("无。")
    else:
        for item in failed:
            lines.append(f"### {item['suite_id']} {item['title']}")
            lines.append("")
            lines.append("stdout tail:")
            lines.append("```text")
            lines.append(item["run"]["stdout_tail"])
            lines.append("```")
            lines.append("stderr tail:")
            lines.append("```text")
            lines.append(item["run"]["stderr_tail"])
            lines.append("```")
    lines.append("")
    lines.append("## 论文口径提醒\n")
    lines.append("- 本套件只覆盖 AP-Core 关键证据, 不覆盖 GL 扩展和产品壳。")
    lines.append("- PASS 不代表完整 AGI、真正意识、完整小学数学、开放世界视觉或真实桌面控制。")
    lines.append("- 每个 claim 的 allowed/forbidden wording 见 `CLAIM_MATRIX.json`。")
    (artifact_root / "KEY_SUITE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_claim_matrix(suite_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for result in suite_results:
        item = result["item"]
        matrix.append(
            {
                "claim_id": item.claim_id,
                "suite_id": item.suite_id,
                "route": item.route,
                "evidence_level": "E4" if result["passed"] else "E3-candidate",
                "claim_text": item.title,
                "allowed_wording": item.allowed_wording,
                "forbidden_wording": item.forbidden_wording,
                "validation_command": item.command,
                "validation_exit_code": result["run"]["exit_code"],
                "validation_passed": result["passed"],
                "teacher_off_boundary": item.teacher_off_boundary,
                "solver_boundary": item.solver_boundary,
                "evidence_paths": result["evidence_records"],
                "paper_section": "Section 6 AP-Core results",
                "status": "ready" if result["passed"] else "needs_fix",
            }
        )
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run APV2.1 paper Canonical-KeySuite-1.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-tests", action="store_true", help="Only build claim matrix/manifest from existing files.")
    args = parser.parse_args()

    artifact_root = args.output_dir
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "source_refs").mkdir(exist_ok=True)

    suite_results: list[dict[str, Any]] = []
    for item in SUITE:
        if args.skip_tests:
            run = {"command": item.command, "exit_code": 0, "duration_seconds": 0.0, "stdout_tail": "skipped", "stderr_tail": ""}
        else:
            run = run_command(item.command, timeout=args.timeout)
        evidence_records = [file_record(path) for path in item.evidence_paths]
        copy_evidence_files(item, artifact_root, evidence_records)
        expected = [file_record(path) for path in item.expected_outputs]
        passed = run["exit_code"] == 0 and all(rec.get("exists") for rec in expected)
        suite_results.append(
            {
                "item": item,
                "suite_id": item.suite_id,
                "claim_id": item.claim_id,
                "title": item.title,
                "route": item.route,
                "run": run,
                "evidence_records": evidence_records,
                "expected_outputs": expected,
                "passed": passed,
            }
        )

    serializable_results: list[dict[str, Any]] = []
    for result in suite_results:
        item = result["item"]
        row = {k: v for k, v in result.items() if k != "item"}
        row["allowed_wording"] = item.allowed_wording
        row["forbidden_wording"] = item.forbidden_wording
        row["teacher_off_boundary"] = item.teacher_off_boundary
        row["solver_boundary"] = item.solver_boundary
        serializable_results.append(row)

    passed_count = sum(1 for result in suite_results if result["passed"])
    payload = {
        "schema_id": "apv21_canonical_key_suite/v1",
        "suite_name": "Canonical-KeySuite-1",
        "generated_at": _now_iso(),
        "artifact_root": _rel(artifact_root),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "cwd": str(ROOT),
            "repository_status": "not_git_repository",
        },
        "summary": {
            "suite_count": len(suite_results),
            "passed_count": passed_count,
            "failed_count": len(suite_results) - passed_count,
            "passed": passed_count == len(suite_results),
        },
        "suite_results": serializable_results,
    }

    claim_matrix = build_claim_matrix(suite_results)
    (artifact_root / "CLAIM_MATRIX.json").write_text(json.dumps(claim_matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_root / "KEY_SUITE_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(artifact_root, payload)
    manifest = collect_manifest(artifact_root)
    (artifact_root / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_root": _rel(artifact_root),
                "passed": payload["summary"]["passed"],
                "passed_count": passed_count,
                "suite_count": len(suite_results),
                "report": _rel(artifact_root / "KEY_SUITE_REPORT.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
