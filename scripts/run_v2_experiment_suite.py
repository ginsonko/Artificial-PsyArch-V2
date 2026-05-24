# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import sys
import shutil
import struct
import tempfile
import wave
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.dataset_runner import run_dataset_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiment_suite"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "chunks").glob("*.summary.jsonl")):
        rows.extend(_iter_jsonl(path))
    rows.sort(key=lambda item: int(item.get("tick_index", -1) or -1))
    return rows


def _sidecar_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "chunks").glob("*.sidecar.jsonl")):
        rows.extend(_iter_jsonl(path))
    rows.sort(key=lambda item: int(item.get("tick_index", -1) or -1))
    return rows


def _metrics_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "chunks").glob("*.metrics.jsonl")):
        rows.extend(_iter_jsonl(path))
    rows.sort(key=lambda item: int(item.get("tick_index", -1) or -1))
    return rows


def _chunk_rows(run_dir: Path, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = f"*.{kind}.jsonl"
    for path in sorted((run_dir / "chunks").glob(pattern)):
        rows.extend(_iter_jsonl(path))
    rows.sort(key=lambda item: int(item.get("tick_index", -1) or -1))
    return rows


def _run_dir(result_row: dict[str, Any]) -> Path:
    result = dict(result_row.get("result", {}) or {})
    clean = str(result.get("run_dir", "") or "")
    if not clean:
        raise RuntimeError("run result missing run_dir")
    return Path(clean)


def _mk_png(label: str, color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (96, 96), color=(12, 12, 12))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 78, 78), radius=8, fill=color)
    draw.text((8, 6), label, fill=(240, 240, 240))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _mk_wav(freq: float, *, duration_sec: float = 0.28) -> bytes:
    sample_rate = 8000
    frames = bytearray()
    for i in range(int(sample_rate * duration_sec)):
        sample = int(12000 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", sample)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def build_text_dataset(root: Path) -> Path:
    dataset_path = root / "text_large_dataset.json"
    texts: list[str] = []
    block = [
        "冬天 的 天气 有点 冷",
        "",
        "冬天 的 天气 有点 冷",
        "我 觉得 今天 有点 冷",
        "",
        "冬天 的 天气 有点 凉",
        "今天 的 温度 有点 凉",
        "",
        "冬天 的 空气 让 人 觉得 冷",
        "这种 感觉 有点 凉",
        "",
        "今天 天气 不错 我 想 出门",
        "如果 下雨 就 带 伞",
        "",
        "今天 天气 不错 我 想 散步",
        "如果 刮风 就 穿 外套",
        "",
    ]
    for _ in range(10):
        texts.extend(block)
    payload = {
        "label": "Phase25 纯文本大型实验",
        "config_overrides": {
            "text_sensor_budget": 28,
            "short_term_successor_tail_limit": 14,
            "memory_candidate_limit": 192,
            "memory_ann_top_k": 64,
            "state_pool_residual_unit_limit": 40,
        },
        "texts": texts,
    }
    _write_json(dataset_path, payload)
    return dataset_path


def build_multimodal_dataset(root: Path) -> Path:
    dataset_path = root / "multimodal_cross_recall_dataset.json"
    red_png = _mk_png("苹果", (220, 50, 40))
    yellow_png = _mk_png("香蕉", (220, 210, 40))
    bell_wav = _mk_wav(440.0)
    low_wav = _mk_wav(220.0)
    items: list[dict[str, Any]] = []
    for _ in range(5):
        items.extend(
            [
                {"text": "红 苹果 叫", "image_b64": _b64(red_png), "audio_b64": _b64(bell_wav), "source_type": "multimodal_experiment"},
                {"text": "黄 香蕉 响", "image_b64": _b64(yellow_png), "audio_b64": _b64(low_wav), "source_type": "multimodal_experiment"},
                {"text": "红 苹果", "image_b64": _b64(red_png), "source_type": "multimodal_experiment"},
                {"text": "叫", "audio_b64": _b64(bell_wav), "source_type": "multimodal_experiment"},
                {"text": "黄 香蕉", "image_b64": _b64(yellow_png), "source_type": "multimodal_experiment"},
                {"text": "响", "audio_b64": _b64(low_wav), "source_type": "multimodal_experiment"},
            ]
        )
    payload = {
        "label": "Phase25 多模态跨模态召回实验",
        "config_overrides": {
            "text_sensor_budget": 20,
            "vision_patch_budget": 20,
            "hearing_window_budget": 12,
            "memory_candidate_limit": 192,
            "memory_ann_top_k": 64,
        },
        "mode": "multimodal",
        "items": items,
    }
    _write_json(dataset_path, payload)
    return dataset_path


def build_action_dataset(root: Path) -> Path:
    dataset_path = root / "action_learning_dataset.json"
    payload = {
        "label": "Phase25 行动奖惩学习实验",
        "config_overrides": {
            "executor_enabled": True,
            "executor_dry_run": True,
            "executor_screenshot_enabled": False,
            "autonomous_capture_required": False,
            "autonomous_auto_feedback_enabled": False,
            "autonomous_teacher_enabled": False,
            "autonomous_llm_gate_enabled": False,
            "text_sensor_budget": 18,
            "short_term_successor_tail_limit": 10,
            "state_pool_residual_unit_limit": 24,
        },
        "runs": [
            {
                "mode": "autonomous_run",
                "label": "海豚训练_记事本",
                "ticks": 18,
                "text_hint": "打开 记事本",
                "reward_schedule": [{"tick_index": i, "reward": 0.32, "punishment": 0.0} for i in range(18)],
            },
            {
                "mode": "autonomous_run",
                "label": "海豚训练_计算器",
                "ticks": 18,
                "text_hint": "打开 计算器",
                "reward_schedule": [{"tick_index": i, "reward": 0.24, "punishment": 0.0} for i in range(18)],
            },
        ],
    }
    _write_json(dataset_path, payload)
    return dataset_path


def build_scale_dataset(root: Path) -> Path:
    dataset_path = root / "scale_stress_dataset.json"
    block = [
        "今天 天气 不错 我 想 出门",
        "如果 下雨 就 带 伞",
        "冬天 的 天气 有点 冷",
        "这种 风 吹 起来 有点 凉",
        "我 还 想 继续 看看",
        "现在 先 不 说 话",
        "",
    ]
    texts: list[str] = []
    for _ in range(60):
        texts.extend(block)
    payload = {
        "label": "Phase25 规模压力实验",
        "config_overrides": {
            "text_sensor_budget": 28,
            "memory_candidate_limit": 224,
            "memory_ann_top_k": 72,
            "state_pool_residual_unit_limit": 48,
            "short_term_memory_limit": 96,
        },
        "texts": texts,
    }
    _write_json(dataset_path, payload)
    return dataset_path


def _action_rules_payload() -> dict[str, Any]:
    return {
        "schema_id": "innate_rules_v2",
        "schema_version": "1.0",
        "rules": [
            {
                "rule_id": "rule::cmd_notepad_drive",
                "enabled": True,
                "priority": 160,
                "display_name": "打开记事本时推动 type_text",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::打开记事本", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::type_text",
                        "reason": "命令匹配_记事本",
                        "params": {"text": "hello notepad"},
                        "formula": {"kind": "constant", "value": 0.96},
                    }
                ],
            },
            {
                "rule_id": "rule::cmd_calc_drive",
                "enabled": True,
                "priority": 158,
                "display_name": "打开计算器时推动 press_key",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::打开计算器", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::press_key",
                        "reason": "命令匹配_计算器",
                        "params": {"key": "enter"},
                        "formula": {"kind": "constant", "value": 0.92},
                    }
                ],
            },
            {
                "rule_id": "rule::prediction_mismatch_dissonance",
                "enabled": True,
                "priority": 150,
                "display_name": "预测落空提升违和感",
                "family": "cognitive_feeling",
                "conditions": [{"metric": "state.prediction_mismatch_mass", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "dissonance",
                        "formula": {"kind": "mul", "metric": "state.prediction_mismatch_mass", "factor": 0.6, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "预测与实际不一致，违和感上升"},
                ],
            },
            {
                "rule_id": "rule::prediction_match_correctness",
                "enabled": True,
                "priority": 148,
                "display_name": "预测命中提升正确感",
                "family": "cognitive_feeling",
                "conditions": [{"metric": "state.prediction_match_mass", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "correctness",
                        "formula": {"kind": "mul", "metric": "state.prediction_match_mass", "factor": 0.35, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "预测命中，正确感上升"},
                ],
            },
            {
                "rule_id": "rule::feedback_reward_expectation",
                "enabled": True,
                "priority": 146,
                "display_name": "奖励信号提升期待",
                "family": "emotion",
                "conditions": [{"metric": "feedback.reward", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "expectation",
                        "formula": {"kind": "mul", "metric": "feedback.reward", "factor": 1.0, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "奖励信号抬升期待"},
                ],
            },
            {
                "rule_id": "rule::feedback_punishment_pressure",
                "enabled": True,
                "priority": 144,
                "display_name": "惩罚信号提升压力",
                "family": "emotion",
                "conditions": [{"metric": "feedback.punishment", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "pressure",
                        "formula": {"kind": "mul", "metric": "feedback.punishment", "factor": 1.0, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "惩罚信号抬升压力"},
                ],
            },
        ],
    }


def _action_tuner_payload() -> dict[str, Any]:
    return {
        "schema_id": "auto_tuner_v2",
        "schema_version": "1.0",
        "enabled": True,
        "profiles": [
            {
                "profile_id": "exp_action_bias_default",
                "enabled": True,
                "display_name": "动作学习基线",
                "when": [{"metric": "metrics.logic_ms", "op": "<", "value": 500.0}],
                "adjustments": [
                    {"target": "attention.focus_gain", "value": 1.05},
                    {"target": "sampling.increment_budget", "value": 18.0},
                    {"target": "prediction.successor_bias_gain", "value": 1.12},
                ],
            }
        ],
    }


@contextmanager
def _override_rules_and_tuner(app: ObservatoryV2App, *, rules_payload: dict[str, Any] | None = None, tuner_payload: dict[str, Any] | None = None):
    original_rules = copy.deepcopy(app.get_rules_payload())
    original_tuner = copy.deepcopy(app.get_tuner_payload())
    try:
        if rules_payload is not None:
            app.save_rules_payload(copy.deepcopy(rules_payload))
        if tuner_payload is not None:
            app.save_tuner_payload(copy.deepcopy(tuner_payload))
        yield
    finally:
        app.save_rules_payload(original_rules)
        app.save_tuner_payload(original_tuner)


def _run_dataset(dataset_path: Path, outputs_root: Path, *, app: ObservatoryV2App | None = None) -> dict[str, Any]:
    effective_dataset_path = dataset_path
    temp_path: Path | None = None
    if app is not None:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "config_overrides" in payload:
            payload = dict(payload)
            payload.pop("config_overrides", None)
            temp_path = outputs_root / "_tmp_dataset_without_config_overrides.json"
            _write_json(temp_path, payload)
            effective_dataset_path = temp_path
    return run_dataset_file(
        effective_dataset_path,
        default_label="Phase25 实验套件",
        timeout_sec=1800.0,
        app=app,
        repo_root_value=REPO_ROOT,
        outputs_root_override=str(outputs_root),
    )


def _make_app(outputs_root: Path, config_overrides: dict[str, Any] | None = None) -> ObservatoryV2App:
    return ObservatoryV2App(
        config=load_config(overrides=config_overrides or None),
        repo_root_value=REPO_ROOT,
        outputs_root_override=str(outputs_root),
    )


def evaluate_text_run(run_dir: Path) -> dict[str, Any]:
    summaries = _summary_rows(run_dir)
    sidecars = _sidecar_rows(run_dir)
    competition_rows = _chunk_rows(run_dir, "competition")
    readable_focus: list[str] = []
    mismatch_examples: list[dict[str, Any]] = []
    phrase_hit_total = 0
    dynamic_phrase_hits = 0
    phrase_labels: set[str] = set()
    synonym_bridge_hits = 0
    synonym_ticks: list[dict[str, Any]] = []
    for summary, sidecar, competition_row in zip(summaries, sidecars, competition_rows):
        focus_units = list(summary.get("a_focus_preview", []) or [])
        if len(focus_units) >= 2:
            readable_focus.append("".join(str(item or "") for item in focus_units))
        competition_packet = dict(competition_row or {})
        competition_summary = dict(competition_packet.get("competition_summary", {}) or {})
        phrase_hit_total += int(competition_summary.get("phrase_hit_count", 0) or 0)
        for item in competition_packet.get("sa_items", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("family", "") or "") == "learned_text_phrase":
                dynamic_phrase_hits += 1
                phrase_labels.add(str(item.get("sa_label", "") or ""))
        state_pool_summary = dict(sidecar.get("state_pool_snapshot", {}) or {})
        prediction_trace = dict(state_pool_summary.get("prediction_trace", {}) or {})
        if float(prediction_trace.get("mismatch_mass", 0.0) or 0.0) > 0.0:
            example = {
                "tick_index": summary.get("tick_index", -1),
                "input_preview": summary.get("input_preview", ""),
                "predicted_texts": prediction_trace.get("predicted_texts", []),
                "actual_texts": prediction_trace.get("actual_texts", []),
                "missed": prediction_trace.get("missed_predicted_labels", []),
                "unexpected": prediction_trace.get("unexpected_labels", []),
            }
            mismatch_examples.append(example)
            unexpected_text = " ".join(str(item) for item in (prediction_trace.get("unexpected_labels", []) or []))
            missed_text = " ".join(str(item) for item in (prediction_trace.get("missed_predicted_labels", []) or []))
            if ("冷" in missed_text and "凉" in unexpected_text) or ("凉" in missed_text and "冷" in unexpected_text):
                synonym_bridge_hits += 1
                synonym_ticks.append(example)
    lexical_overlap = 0.0
    if len(readable_focus) >= 2:
        pairs = 0
        total = 0
        for left, right in zip(readable_focus, readable_focus[1:]):
            total += len(set(left) & set(right))
            pairs += 1
        lexical_overlap = total / max(1, pairs)
    return {
        "tick_count": len(summaries),
        "focus_chain_readable_count": len(readable_focus),
        "focus_chain_examples": readable_focus[:16],
        "mismatch_tick_count": len(mismatch_examples),
        "mismatch_examples": mismatch_examples[:10],
        "synonym_bridge_hits": synonym_bridge_hits,
        "synonym_examples": synonym_ticks[:6],
        "phrase_hit_total": phrase_hit_total,
        "dynamic_phrase_hit_count": dynamic_phrase_hits,
        "dynamic_phrase_labels": sorted(phrase_labels)[:24],
        "mean_adjacent_lexical_overlap": round(float(lexical_overlap), 4),
        "passed_language_chain": len(readable_focus) >= max(10, len(summaries) // 5),
        "passed_synonym_mismatch_signal": synonym_bridge_hits >= 1 and len(mismatch_examples) >= 2,
        "passed_abstraction_signal": dynamic_phrase_hits >= 6 and len(phrase_labels) >= 3,
    }


def evaluate_multimodal_run(run_dir: Path) -> dict[str, Any]:
    summaries = _summary_rows(run_dir)
    sidecars = _sidecar_rows(run_dir)
    exact_memory_rows = _chunk_rows(run_dir, "exactmem")
    cross_modal_memory_ticks = 0
    cross_modal_candidate_ticks = 0
    true_cross_modal_hits = 0
    examples: list[dict[str, Any]] = []
    for summary, sidecar, exact_memory_row in zip(summaries, sidecars, exact_memory_rows):
        exact_memory = dict(exact_memory_row or {})
        modalities = list(exact_memory.get("modalities", []) or [])
        if len(modalities) >= 2:
            cross_modal_memory_ticks += 1
        bn_preview = list(summary.get("bn_preview", []) or [])
        for item in bn_preview:
            channels = set(str(x or "") for x in (item.get("candidate_sources", []) or []) if str(x or ""))
            text = str(item.get("text", "") or "")
            if "vector_ann" in channels or "spacetime" in channels or "recent_window" in channels:
                cross_modal_candidate_ticks += 1
            memory_modalities = []
            for row in sidecar.get("bn_list", []) or []:
                if str(row.get("memory_id", "") or "") == str(item.get("memory_id", "") or ""):
                    memory_modalities = list((row.get("memory_modalities", []) or row.get("modalities", []) or []))
                    break
            if len(set(memory_modalities) - set(modalities)) >= 1 or (len(memory_modalities) >= 3 and len(modalities) < len(memory_modalities)):
                true_cross_modal_hits += 1
                break
        if len(examples) < 10:
            examples.append(
                {
                    "tick_index": summary.get("tick_index", -1),
                    "input_preview": summary.get("input_preview", ""),
                    "modalities": modalities,
                    "bn_preview": [
                        {
                            "memory_id": item.get("memory_id", ""),
                            "text": item.get("text", ""),
                            "candidate_sources": item.get("candidate_sources", []),
                        }
                        for item in bn_preview[:3]
                    ],
                }
            )
    return {
        "tick_count": len(summaries),
        "multi_modal_memory_ticks": cross_modal_memory_ticks,
        "cross_modal_candidate_ticks": cross_modal_candidate_ticks,
        "true_cross_modal_hits": true_cross_modal_hits,
        "examples": examples,
        "passed_cross_modal_binding": cross_modal_memory_ticks >= 4 and true_cross_modal_hits >= 4,
    }


def evaluate_action_runs(run_payload: dict[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for row in run_payload.get("runs", []) or []:
        run_dir = _run_dir(row)
        sidecars = _sidecar_rows(run_dir)
        selected_names: list[str] = []
        merged_rewards = 0.0
        merged_punishments = 0.0
        teacher_rewards = 0.0
        teacher_punishments = 0.0
        bias_tail: list[dict[str, Any]] = []
        context_bias_tail: list[dict[str, Any]] = []
        for sidecar in sidecars:
            sandbox = dict(sidecar.get("sandbox_result", {}) or {})
            for action in sandbox.get("selected_actions", []) or []:
                if not isinstance(action, dict):
                    continue
                name = str(action.get("action_name", "") or "")
                if name:
                    selected_names.append(name)
            feedback_used = dict((sidecar.get("autonomous_sidecar", {}) or {}).get("feedback_used", {}) or {})
            merged_rewards += float(feedback_used.get("reward", 0.0) or 0.0)
            merged_punishments += float(feedback_used.get("punishment", 0.0) or 0.0)
            teacher_feedback = dict(sidecar.get("teacher_feedback", {}) or {})
            teacher_rewards += float(teacher_feedback.get("reward", 0.0) or 0.0)
            teacher_punishments += float(teacher_feedback.get("punishment", 0.0) or 0.0)
            bias_tail = list(sidecar.get("action_learning_bias_summary", []) or [])
            context_bias_tail = list(sidecar.get("action_learning_context_bias_summary", []) or [])
        dominant_action = max(set(selected_names), key=selected_names.count) if selected_names else ""
        label = str(row.get("label", "") or "")
        expected_action = "type_text" if "记事本" in label else "press_key" if "计算器" in label else dominant_action
        reports.append(
            {
                "label": label,
                "run_id": str((row.get("result", {}) or {}).get("run_id", "") or ""),
                "expected_action": expected_action,
                "dominant_action": dominant_action,
                "selected_action_count": len(selected_names),
                "selected_actions_preview": selected_names[:12],
                "feedback_reward_total": round(merged_rewards, 4),
                "feedback_punishment_total": round(merged_punishments, 4),
                "teacher_reward_total": round(teacher_rewards, 4),
                "teacher_punishment_total": round(teacher_punishments, 4),
                "bias_tail": bias_tail[:8],
                "context_bias_tail": context_bias_tail[:8],
                "passed": dominant_action == expected_action and merged_rewards > merged_punishments,
            }
        )
    return {
        "run_reports": reports,
        "passed_action_learning": all(bool(item.get("passed", False)) for item in reports) if reports else False,
    }


def evaluate_emotion_on_action_runs(run_payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in run_payload.get("runs", []) or []:
        run_dir = _run_dir(run)
        sidecars = _sidecar_rows(run_dir)
        dissonance_values: list[float] = []
        correctness_values: list[float] = []
        expectation_values: list[float] = []
        pressure_values: list[float] = []
        reward_values: list[float] = []
        punishment_values: list[float] = []
        rule_logs: list[str] = []
        for sidecar in sidecars:
            emotion = dict((sidecar.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
            dissonance_values.append(float(emotion.get("dissonance", 0.0) or 0.0))
            correctness_values.append(float(emotion.get("correctness", 0.0) or 0.0))
            expectation_values.append(float(emotion.get("expectation", 0.0) or 0.0))
            pressure_values.append(float(emotion.get("pressure", 0.0) or 0.0))
            feedback = dict((sidecar.get("autonomous_sidecar", {}) or {}).get("feedback_used", {}) or {})
            reward_values.append(float(feedback.get("reward", 0.0) or 0.0))
            punishment_values.append(float(feedback.get("punishment", 0.0) or 0.0))
            for log in (sidecar.get("rules_result", {}) or {}).get("rule_logs", []) or []:
                if isinstance(log, dict):
                    rule_logs.append(str(log.get("message", "") or ""))
        rows.append(
            {
                "label": run.get("label", ""),
                "mean_dissonance": round(sum(dissonance_values) / max(1, len(dissonance_values)), 4),
                "mean_correctness": round(sum(correctness_values) / max(1, len(correctness_values)), 4),
                "mean_expectation": round(sum(expectation_values) / max(1, len(expectation_values)), 4),
                "mean_pressure": round(sum(pressure_values) / max(1, len(pressure_values)), 4),
                "reward_total": round(sum(reward_values), 4),
                "punishment_total": round(sum(punishment_values), 4),
                "rule_log_preview": rule_logs[:12],
            }
        )
    expectation_advantage = sum(float(item.get("mean_expectation", 0.0) or 0.0) for item in rows) - sum(float(item.get("mean_pressure", 0.0) or 0.0) for item in rows)
    correctness_advantage = sum(float(item.get("mean_correctness", 0.0) or 0.0) for item in rows) - sum(float(item.get("mean_dissonance", 0.0) or 0.0) for item in rows)
    return {
        "rows": rows,
        "expectation_minus_pressure_total": round(expectation_advantage, 4),
        "correctness_minus_dissonance_total": round(correctness_advantage, 4),
        "passed_emotion_signal": expectation_advantage >= 0.2 and correctness_advantage >= 0.0,
    }


def evaluate_scale_run(run_dir: Path) -> dict[str, Any]:
    metrics = _metrics_rows(run_dir)
    logic_values = [float(item.get("logic_ms", 0.0) or 0.0) for item in metrics]
    return {
        "tick_count": len(metrics),
        "logic_ms_mean": round(sum(logic_values) / max(1, len(logic_values)), 4) if logic_values else 0.0,
        "logic_ms_max": round(max(logic_values), 4) if logic_values else 0.0,
        "passed_scale_smoke": bool(metrics) and max(logic_values, default=0.0) < 1500.0,
    }


def run_suite(output_root: Path, *, clean: bool) -> dict[str, Any]:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = output_root / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    text_dataset = build_text_dataset(dataset_root)
    multimodal_dataset = build_multimodal_dataset(dataset_root)
    action_dataset = build_action_dataset(dataset_root)
    scale_dataset = build_scale_dataset(dataset_root)

    text_result = _run_dataset(text_dataset, output_root / "text")
    multimodal_result = _run_dataset(multimodal_dataset, output_root / "multimodal")

    action_outputs = output_root / "action"
    action_dataset_payload = json.loads(action_dataset.read_text(encoding="utf-8"))
    action_config_overrides = dict(action_dataset_payload.get("config_overrides", {}) or {})
    action_app = _make_app(action_outputs, config_overrides=action_config_overrides)
    with _override_rules_and_tuner(action_app, rules_payload=_action_rules_payload(), tuner_payload=_action_tuner_payload()):
        action_result = _run_dataset(action_dataset, action_outputs, app=action_app)

    scale_result = _run_dataset(scale_dataset, output_root / "scale")

    text_eval = evaluate_text_run(_run_dir(text_result["runs"][0]))
    multimodal_eval = evaluate_multimodal_run(_run_dir(multimodal_result["runs"][0]))
    action_eval = evaluate_action_runs(action_result)
    emotion_eval = evaluate_emotion_on_action_runs(action_result)
    scale_eval = evaluate_scale_run(_run_dir(scale_result["runs"][0]))

    report = {
        "schema_id": "phase25_experiment_report/v2",
        "schema_version": "1.0",
        "output_root": str(output_root),
        "experiments": {
            "text_large": {"dataset": str(text_dataset), "result": text_result, "evaluation": text_eval},
            "multimodal_small": {"dataset": str(multimodal_dataset), "result": multimodal_result, "evaluation": multimodal_eval},
            "action_learning": {"dataset": str(action_dataset), "result": action_result, "evaluation": action_eval},
            "emotion_learning": {"derived_from": "action_learning", "evaluation": emotion_eval},
            "scale_stress": {"dataset": str(scale_dataset), "result": scale_result, "evaluation": scale_eval},
        },
        "summary": {
            "text_passed_language_chain": bool(text_eval.get("passed_language_chain", False)),
            "text_passed_synonym_mismatch_signal": bool(text_eval.get("passed_synonym_mismatch_signal", False)),
            "text_passed_abstraction_signal": bool(text_eval.get("passed_abstraction_signal", False)),
            "multimodal_passed_cross_modal_binding": bool(multimodal_eval.get("passed_cross_modal_binding", False)),
            "action_passed_learning": bool(action_eval.get("passed_action_learning", False)),
            "emotion_passed_signal": bool(emotion_eval.get("passed_emotion_signal", False)),
            "scale_passed_smoke": bool(scale_eval.get("passed_scale_smoke", False)),
        },
    }
    report_path = output_root / "phase25_experiment_report.json"
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="AP V2 大型实验套件")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="实验输出目录")
    parser.add_argument("--clean", action="store_true", help="运行前清空输出目录")
    args = parser.parse_args()
    report = run_suite(Path(args.output_root), clean=bool(args.clean))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
