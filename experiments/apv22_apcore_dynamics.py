from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.defaults import RuntimeConfig, ShortTermSlotConfig
from core.runtime.engine import APV21Runtime
from memory.persistence import RecordingMemoryPersistence
from memory.spacetime.transition_store import TransitionStore
from memory.store import MemoryStore
from core.action import ActionConsequencePlanner, TextActionActuator
from core.emotion.emotion_modulator import EmotionModulator


def _round4(value: float) -> float:
    return round(float(value), 4)


def _labels(rows: list[dict]) -> list[str]:
    return [str(row.get("sa_label", "") or "") for row in rows or [] if str(row.get("sa_label", "") or "")]


def _item(label: str, *, real: float = 1.0, virtual: float = 0.0, source_type: str = "external_text") -> dict:
    token = str(label).split("::")[-1]
    return {
        "sa_label": str(label),
        "display_text": token,
        "family": "text",
        "source_type": str(source_type),
        "real_energy": float(real),
        "virtual_energy": float(virtual),
        "cognitive_pressure": float(real - virtual),
    }


def _memory(
    *,
    recall_top_k: int = 4,
    predict_top_k: int = 4,
    persistence=None,
    store_cls=MemoryStore,
) -> MemoryStore:
    return store_cls(
        recall_top_k=recall_top_k,
        predict_top_k=predict_top_k,
        prediction_energy_scale=0.72,
        max_snapshots_per_kind=128,
        candidate_limit=64,
        scoring_candidate_limit=64,
        online_enabled=True,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        transition_learned_weight=0.18,
        persistence=persistence,
    )


def _write(memory: MemoryStore, tick: int, labels: list[str], source_text: str, *, memory_kind: str = "state") -> dict:
    return memory.write_snapshot(
        tick_index=int(tick),
        memory_kind=memory_kind,
        items=[_item(label) for label in labels],
        focus_labels=[],
        source_text=source_text,
    )


def _pass(passed: bool, partial: bool = False) -> str:
    if passed:
        return "pass"
    if partial:
        return "partial"
    return "fail"


class AblatedNegativeFeedbackMemoryStore(MemoryStore):
    def _is_negative_feedback_text_payload_item(self, row: dict) -> bool:
        return False


def _pressure_planner() -> ActionConsequencePlanner:
    planner = ActionConsequencePlanner(
        enabled=True,
        selection_threshold=0.32,
        max_selected_actions=4,
        fatigue_decay=0.84,
        fatigue_step=0.0,
        bias_learning_rate=0.20,
        bias_gain=0.32,
        confidence_gain=0.18,
        wait_base_drive=0.18,
    )
    selected = [
        {
            "action_id": "action::text_commit",
            "actuator_id": "actuator::text_editor",
            "predicted_outcome": {
                "reward": 0.08,
                "punishment": 0.03,
                "correctness": 0.10,
                "pressure": 0.04,
                "confidence": 0.58,
            },
        }
    ]
    for _ in range(12):
        planner.record_feedback(
            selected_actions=selected,
            observed_feedback={"reward": 0.12, "punishment": 0.0, "correctness": 0.18, "confidence": 0.74},
        )
    return planner


def _pressure_draft_items(text: str = "OK") -> list[dict]:
    actuator = TextActionActuator(max_visible_buffer=12)
    for offset, token in enumerate(text, start=1):
        actuator.step(
            tick_index=offset,
            input_text="",
            selected_actions=[{"action_id": "action::text_insert", "params": {"token": token, "reason": "probe"}}],
            fast_cn=[],
            slow_cn=[],
            focus_labels=[],
            cognitive_feelings={"channels": {}},
        )
    actuator.step(
        tick_index=len(text) + 1,
        input_text="",
        selected_actions=[{"action_id": "action::text_reread", "params": {"span": [0, len(text)]}}],
        fast_cn=[],
        slow_cn=[],
        focus_labels=[],
        cognitive_feelings={"channels": {}},
    )
    return actuator.short_term_context_items()


def _pressure_trace(level: float = 0.0, gap: float = 0.0, anchor: float = 0.0) -> dict:
    anchors = []
    if anchor > 0.0:
        anchors = [
            {
                "anchor_type": "pressure",
                "anchor_id": "pressure-anchor-1",
                "level": float(anchor),
                "expected_punishment": 0.6,
                "expected_pressure": float(anchor),
                "source_memory_id": "mem-x",
            }
        ]
    return {
        "channels": {"pressure_level": float(level), "expectation_gap": float(gap)},
        "anchor_verification": {"anchors": anchors, "verified": anchors, "missed": []},
    }


def _pressure_commit_ready_item(commit_ready: dict) -> dict:
    return {
        "sa_label": "state::commit_ready",
        "display_text": "commit_ready",
        "family": "state",
        "source_type": "text_action",
        "real_energy": float(commit_ready.get("commit_readiness", 0.0) or 0.0),
        "virtual_energy": float(commit_ready.get("commit_reread_need", 0.0) or 0.0) * 0.42,
        "cognitive_pressure": float(commit_ready.get("commit_readiness", 0.0) or 0.0),
        "anchor_meta": {
            **dict(commit_ready),
            "schema_id": "text_commit_readiness_state/v1",
            "policy": "short_lived_commit_readiness_is_learnable_state_not_force_submit",
        },
    }


def _pressure_mismatch_item() -> dict:
    return {
        "sa_label": "text_action::write::K",
        "display_text": "K",
        "family": "text_action",
        "source_type": "text_action",
        "real_energy": 0.0,
        "virtual_energy": 0.0,
        "cognitive_pressure": 0.0,
        "anchor_meta": {
            "schema_id": "draft_read_token/v1",
            "event_type": "write_mismatch",
            "token": "K",
            "feedback_outcome": "punished",
            "feedback_punishment": 0.62,
            "feedback_reward": 0.0,
            "feedback_correctness": 0.0,
            "feedback_reference_token": "K",
            "feedback_expected_token": "K",
            "feedback_token_mismatch": True,
            "used_in_strict_teacher_off_input": False,
        },
    }


def _run_pressure_dynamics_case(*, name: str, pressure: float, gap: float, anchor: float, include_mismatch: bool) -> dict:
    planner = _pressure_planner()
    modulator = EmotionModulator()
    if pressure < 0.2:
        for _ in range(5):
            modulator.update(
                cognitive_feelings={"channels": {"correctness": 0.7, "coherence": 0.6, "fulfillment": 0.4, "expectation": 0.2}},
                reward=0.7,
                punishment=0.0,
            )
    else:
        for _ in range(5):
            modulator.update(
                cognitive_feelings={"channels": {"pressure": pressure, "dissonance": pressure * 0.75, "uncertainty": 0.22, "correctness": 0.18}},
                reward=0.0,
                punishment=0.62,
            )

    state_items = _pressure_draft_items("OK")
    fast_cn = [
        {
            "predicted_items": [
                {"sa_label": "text::K", "display_text": "K", "virtual_energy": 0.81},
                {"sa_label": "text::Q", "display_text": "Q", "virtual_energy": 0.10},
            ]
        }
    ]
    if include_mismatch:
        state_items.append(_pressure_mismatch_item())
    commit_ready = planner.draft_commit_readiness_context(
        state_items,
        current_tick=10,
        fast_cn=fast_cn,
        slow_cn=[],
        correctness=0.52 if pressure < 0.2 else 0.24,
        grasp=0.44,
        pressure=pressure,
        dissonance=pressure * 0.7,
        uncertainty=0.12,
        pressure_anchor_level=anchor,
        expectation_gap=gap,
    )
    state_items = list(state_items) + [_pressure_commit_ready_item(commit_ready)]

    trace = planner.plan(
        tick_index=11,
        state_snapshot_items=state_items,
        fast_bn=[],
        fast_cn=fast_cn,
        slow_bn=[],
        slow_cn=[],
        cognitive_feelings={
            "channels": {
                "correctness": 0.52 if pressure < 0.2 else 0.24,
                "grasp": 0.44,
                "pressure": pressure,
                "dissonance": pressure * 0.7,
                "uncertainty": 0.12,
                "fulfillment": 0.25,
                "task_available": 0.4,
            }
        },
        rhythm_trace={"channels": {"phase_expectation": 0.2}},
        time_trace={"channels": {"confidence": 0.2}},
        expectation_pressure_trace=_pressure_trace(level=pressure, gap=gap, anchor=anchor),
        emotion_modulation=modulator.get_modulation(),
    )
    candidates = {str(row.get("action_id", "") or ""): dict(row) for row in list(trace.get("candidates", []) or [])}
    selected = [str(row.get("action_id", "") or "") for row in list(trace.get("selected_actions", []) or [])]
    goal_alignment = dict(commit_ready.get("goal_alignment", {}) or {})
    satisfaction_field = dict(commit_ready.get("satisfaction_field", {}) or {})
    return {
        "name": name,
        "pressure": _round4(float(pressure)),
        "pressure_anchor": _round4(float(anchor)),
        "expectation_gap": _round4(float(gap)),
        "effective_threshold": _round4(float(trace.get("effective_threshold", 0.0) or 0.0)),
        "selected_actions": selected,
        "commit_readiness": _round4(float(commit_ready.get("commit_readiness", 0.0) or 0.0)),
        "commit_reread_need": _round4(float(commit_ready.get("commit_reread_need", 0.0) or 0.0)),
        "goal_alignment": _round4(float(goal_alignment.get("goal_alignment", 0.0) or 0.0)),
        "goal_alignment_block": goal_alignment,
        "satisfaction_field": satisfaction_field,
        "satisfaction": _round4(float(commit_ready.get("draft_eval", {}).get("satisfaction", 0.0) or 0.0)),
        "candidate_drives": {
            action_id: _round4(float(candidates.get(action_id, {}).get("drive", 0.0) or 0.0))
            for action_id in ("action::text_commit", "action::text_reread", "action::text_replace", "action::replay_episode")
        },
        "candidate_notes": {
            action_id: list(candidates.get(action_id, {}).get("notes", []) or [])[:10]
            for action_id in ("action::text_commit", "action::text_reread", "action::text_replace", "action::replay_episode")
        },
    }


def _feedback_text_item(token: str, *, reward: float, punishment: float, correctness: float, tick: int) -> dict:
    outcome = "punished" if punishment > max(reward, correctness) else "rewarded"
    return {
        **_item(f"text::{token}", real=0.40, virtual=0.08, source_type="internal_draft_read"),
        "last_seen_tick": int(tick),
        "anchor_meta": {
            "schema_id": "draft_read_token/v1",
            "event_type": "draft_read_token",
            "token": str(token),
            "feedback_outcome": outcome,
            "feedback_reward": float(reward),
            "feedback_punishment": float(punishment),
            "feedback_correctness": float(correctness),
            "current_read_tick": True,
            "current_glyph_role": "read_tick_target",
            "prediction_payload_priority": "current_glyph_character",
            "used_in_strict_teacher_off_input": False,
        },
    }


def run_feedback_override() -> dict:
    memory = _memory()
    memory.write_snapshot(
        tick_index=1,
        memory_kind="state",
        items=[_feedback_text_item("same", reward=0.72, punishment=0.0, correctness=0.78, tick=1)],
        focus_labels=[],
        source_text="rewarded same",
    )
    memory.write_snapshot(
        tick_index=2,
        memory_kind="state",
        items=[_feedback_text_item("same", reward=0.0, punishment=0.82, correctness=0.0, tick=2)],
        focus_labels=[],
        source_text="punished same",
    )
    snapshot = memory.latest_snapshot("state") or {}
    payload = list(snapshot.get("prediction_payload_items", []) or [])
    payload_labels = _labels(payload)
    repair_rows = [row for row in payload if str(row.get("sa_label", "")).startswith("text_revision_opportunity::negative_feedback::same")]
    passed = "text::same" not in payload_labels and bool(repair_rows)
    return {
        "experiment": "FeedbackOverride-1",
        "verdict": _pass(passed, bool(repair_rows)),
        "design": "Punishment-dominant later evidence overrides a rewarded same-token positive payload by deriving repair context.",
        "observed": {
            "payload_labels_after_punishment": payload_labels,
            "repair_meta": dict((repair_rows[0].get("anchor_meta", {}) if repair_rows else {}) or {}),
        },
        "boundary": _boundary(),
    }


def run_persistence_reload() -> dict:
    recorder = RecordingMemoryPersistence()
    writer = _memory(persistence=recorder)
    _write(writer, 1, ["text::cue", "text::context"], "cue context")
    _write(writer, 2, ["text::outcome"], "outcome")
    _write(
        writer,
        3,
        ["short_term_slot::summary", "short_term_slot::item::text::cue", "short_term_slot::continuity"],
        "slot cue context",
        memory_kind="short_term_slot",
    )
    reader = _memory()
    reader._persistence = recorder
    warm = reader.warm_load_from_persistence(limit_per_kind=8)
    bn = reader.recall([_item("text::cue"), _item("text::context")], memory_kind="state", top_k=2)
    cn = reader.successors(bn[0]["memory_id"], memory_kind="state", top_k=2, source_b_row=bn[0]) if bn else []
    slot_bn = reader.recall([_item("short_term_slot::item::text::cue", virtual=0.5, source_type="short_term_slot")], memory_kind="short_term_slot", top_k=2)
    predicted_labels = [item.get("sa_label") for row in cn for item in row.get("predicted_items", [])]
    passed = bool(warm.get("loaded", 0) >= 3 and bn and "text::outcome" in predicted_labels and slot_bn)
    return {
        "experiment": "PersistenceReload-1",
        "verdict": _pass(passed, bool(warm.get("loaded", 0) >= 3 and bn)),
        "design": "Reloaded authoritative memory restores state B recall, C successor prediction, and short-term-slot memory objects.",
        "observed": {
            "warm_load": warm,
            "write_count": len(recorder.writes),
            "bn_source_texts": [row.get("source_text") for row in bn],
            "predicted_labels": predicted_labels,
            "slot_bn_source_texts": [row.get("source_text") for row in slot_bn],
        },
        "boundary": _boundary(),
    }


def run_negative_feedback_ablation() -> dict:
    normal = _memory()
    ablated = _memory(store_cls=AblatedNegativeFeedbackMemoryStore)
    punished_item = _feedback_text_item("bad", reward=0.0, punishment=0.66, correctness=0.0, tick=7)
    for memory in (normal, ablated):
        memory.write_snapshot(
            tick_index=7,
            memory_kind="state",
            items=[punished_item],
            focus_labels=[],
            source_text="punished bad",
        )
    normal_payload = _labels(list((normal.latest_snapshot("state") or {}).get("prediction_payload_items", []) or []))
    ablated_payload = _labels(list((ablated.latest_snapshot("state") or {}).get("prediction_payload_items", []) or []))
    normal_suppressed = "text::bad" not in normal_payload and any(label.startswith("text_revision_opportunity::negative_feedback::bad") for label in normal_payload)
    ablated_leaks_bad = "text::bad" in ablated_payload and not any(label.startswith("text_revision_opportunity::negative_feedback::bad") for label in ablated_payload)
    passed = bool(normal_suppressed and ablated_leaks_bad)
    return {
        "experiment": "NegativeFeedback-Ablation-1",
        "verdict": _pass(passed, normal_suppressed or ablated_leaks_bad),
        "design": "Disable only the negative-text detector and verify punished raw text leaks back into positive payload.",
        "observed": {
            "normal_payload_labels": normal_payload,
            "ablated_payload_labels": ablated_payload,
            "normal_suppressed": normal_suppressed,
            "ablated_leaks_bad": ablated_leaks_bad,
        },
        "boundary": _boundary(),
    }


def run_short_term_interruption_recovery() -> dict:
    runtime = APV21Runtime()

    def focus_pack(labels: list[str]) -> list[dict]:
        return [
            {
                **_item(label, real=1.0, virtual=0.0, source_type="white_box_focus_probe"),
                "focus_score": 1.0 - idx * 0.05,
                "anchor_meta": {"is_focus": True},
            }
            for idx, label in enumerate(labels)
        ]

    packs = [
        focus_pack(["focus::river", "focus::stone"]),
        focus_pack(["focus::river", "focus::stone", "focus::bright"]),
        focus_pack(["focus::alarm", "focus::red"]),
        focus_pack(["focus::alarm", "focus::red", "focus::loud"]),
        focus_pack(["focus::river", "focus::stone", "focus::bright"]),
    ]
    slot_trace = {}
    for tick, pack in enumerate(packs):
        runtime.focus_buffer.push(pack, tick_index=tick)
        focus_trace = runtime.focus_buffer.trace(tick_index=tick)
        slot_trace = runtime.short_term_slot.build(
            tick_index=tick,
            focus_items=pack,
            focus_continuation_trace=focus_trace,
            short_term_memory_trace={"last_recall": {"available": False, "score": 0.0}},
            rhythm_trace={},
            runtime_load_trace={"channels": {"load_ratio": 0.0}},
        )
        runtime._last_short_term_slot_trace = dict(slot_trace)
        runtime.state_pool.begin_tick(tick)
        runtime.state_pool.apply_external_items(list(slot_trace.get("items", []) or []), tick_index=tick)
        if slot_trace.get("items"):
            runtime.memory.write_snapshot(
                tick_index=tick,
                memory_kind="short_term_slot",
                items=list(slot_trace.get("items", []) or []),
                focus_labels=list(slot_trace.get("focus_labels", []) or []),
                source_text=f"white-box focus tick {tick}",
            )

    dynamics = runtime.focus_buffer.continuity_dynamics(tick_index=len(packs) - 1)
    slot_labels = _labels(list(slot_trace.get("items", []) or []))
    recent_interruptions = list(dynamics.get("recent_interruptions", []) or [])
    recent_resumptions = list(dynamics.get("recent_resumptions", []) or [])

    disabled_runtime = APV21Runtime(config=replace(RuntimeConfig(), short_term_slot=ShortTermSlotConfig(enabled=False)))
    disabled_slot_trace = disabled_runtime.short_term_slot.build(
        tick_index=0,
        focus_items=packs[-1],
        focus_continuation_trace={},
        short_term_memory_trace={},
        rhythm_trace={},
        runtime_load_trace={},
    )
    disabled_slot_labels = _labels(list(disabled_slot_trace.get("items", []) or []))
    slot_virtual_mass = sum(float(row.get("virtual_energy", 0.0) or 0.0) for row in list(slot_trace.get("items", []) or []))
    state_readback = runtime.state_pool.rows_for_labels(slot_labels[:6])
    slot_memory = runtime.memory.latest_snapshot("short_term_slot") or {}
    passed = bool(
        recent_interruptions
        and recent_resumptions
        and slot_labels
        and state_readback
        and _labels(list(slot_memory.get("items", []) or []))
        and not disabled_slot_labels
        and slot_virtual_mass > 0.0
    )
    return {
        "experiment": "ShortTermInterruptionRecovery-1",
        "verdict": _pass(passed, bool(recent_interruptions and slot_labels)),
        "design": "Stable focus, interruption, and return should create observable interruption/resumption traces while slot packets keep narrative energy online.",
        "observed": {
            "recent_interruptions": recent_interruptions,
            "recent_resumptions": recent_resumptions,
            "slot_labels": slot_labels[:20],
            "slot_virtual_mass": _round4(slot_virtual_mass),
            "disabled_slot_labels": disabled_slot_labels[:8],
            "state_readback_labels": _labels(state_readback),
            "slot_memory_labels": _labels(list(slot_memory.get("items", []) or []))[:12],
            "active_episode_id": dynamics.get("active_episode_id"),
            "episode_count": dynamics.get("episode_count"),
        },
        "boundary": _boundary(),
    }


def run_residual_depth() -> dict:
    memory = _memory(recall_top_k=4, predict_top_k=3)
    _write(memory, 1, ["text::A", "text::B"], "AB")
    _write(memory, 2, ["text::C", "text::D"], "CD")
    _write(memory, 3, ["text::E"], "E")
    _write(memory, 4, ["text::A"], "A")
    query = [_item("text::A", real=1.3), _item("text::B", real=1.2), _item("text::C", real=1.1), _item("text::E", real=1.0)]
    residual = memory.recall_residual(query, memory_kind="state", top_k=4)
    ordinary = memory.recall(query, memory_kind="state", top_k=4)
    trace = [dict(row.get("residual_absorption", {}) or {}) for row in residual if row.get("residual_absorption")]
    matched_by_round = [row.get("matched_labels", []) for row in trace]
    winners = [row.get("source_text") for row in residual]
    mass_declines = all(float(row.get("residual_mass_after", 0.0) or 0.0) < float(row.get("residual_mass_before", 0.0) or 0.0) for row in trace)
    covers_e = any("text::E" in set(labels) for labels in matched_by_round) or "E" in winners
    passed = bool(len(trace) >= 3 and mass_declines and covers_e and len(set(winners)) == len(winners))
    return {
        "experiment": "ResidualDepth-1",
        "verdict": _pass(passed, len(trace) >= 2 and mass_declines),
        "design": "A multi-component query should be absorbed over multiple B rounds rather than reduced to a single winner.",
        "observed": {
            "residual_winners": winners,
            "ordinary_winners": [row.get("source_text") for row in ordinary],
            "absorption_trace": trace,
            "matched_by_round": matched_by_round,
            "mass_declines": mass_declines,
        },
        "boundary": _boundary(),
    }


def run_successor_peak_ablation() -> dict:
    store = TransitionStore()
    store.register_snapshot({"memory_id": "m0", "memory_kind": "state", "tick_index": 0, "items": []})
    for lag, label in [(1, "text::one"), (2, "text::two"), (5, "text::five")]:
        store.register_snapshot(
            {
                "memory_id": f"m{lag}",
                "memory_kind": "state",
                "tick_index": lag,
                "prediction_payload_items": [_item(label, real=1.0)],
            }
        )
        store.link_successor("state", "m0", f"m{lag}")
    shaped = store.successors("state", "m0", top_k=3, prediction_energy_scale=1.0, lag_shaping_enabled=True)
    flat = store.successors("state", "m0", top_k=3, prediction_energy_scale=1.0, lag_shaping_enabled=False)
    shaped_kernels = [float(row.get("successor_lag_kernel", 0.0) or 0.0) for row in shaped]
    flat_kernels = [float(row.get("successor_lag_kernel", 0.0) or 0.0) for row in flat]
    passed = bool(shaped_kernels[0] == 1.0 and shaped_kernels[0] > shaped_kernels[1] > shaped_kernels[2] and flat_kernels == [1.0, 1.0, 1.0])
    return {
        "experiment": "SuccessorPeakAblation-1",
        "verdict": _pass(passed, bool(shaped_kernels and flat_kernels)),
        "design": "Lag-shaped successor energy should have a next-tick peak; disabling lag shaping should flatten the distribution.",
        "observed": {
            "shaped": [
                {
                    "successor_memory_id": row.get("successor_memory_id"),
                    "lag": row.get("successor_lag_ticks"),
                    "kernel": row.get("successor_lag_kernel"),
                    "predicted_labels": _labels(list(row.get("predicted_items", []) or [])),
                }
                for row in shaped
            ],
            "flat_kernels": flat_kernels,
        },
        "boundary": _boundary(),
    }


def run_double_energy_balance_pressure_dynamics() -> dict:
    """
    A10: verify that cognitive pressure pushes the planner toward reread /
    revise / replay and away from direct commit, while the effect weakens when
    mismatch or pressure anchors are removed.
    """
    cases = [
        _run_pressure_dynamics_case(name="baseline", pressure=0.04, gap=0.0, anchor=0.0, include_mismatch=False),
        _run_pressure_dynamics_case(name="stress", pressure=0.78, gap=0.0, anchor=0.82, include_mismatch=True),
        _run_pressure_dynamics_case(name="stress_no_anchor", pressure=0.78, gap=0.0, anchor=0.0, include_mismatch=True),
        _run_pressure_dynamics_case(name="stress_no_mismatch", pressure=0.78, gap=0.0, anchor=0.82, include_mismatch=False),
    ]
    baseline, stress, stress_no_anchor, stress_no_mismatch = cases
    checks = {
        "baseline_commit_has_positive_drive": baseline["candidate_drives"]["action::text_commit"] > 0.0,
        "stress_commit_drops_to_zero_or_near_zero": stress["candidate_drives"]["action::text_commit"] <= 0.01,
        "stress_reread_or_replace_outruns_commit": max(
            stress["candidate_drives"]["action::text_reread"],
            stress["candidate_drives"]["action::text_replace"],
            stress["candidate_drives"]["action::replay_episode"],
        ) > stress["candidate_drives"]["action::text_commit"],
        "stress_anchor_boosts_replay": stress["candidate_drives"]["action::replay_episode"] > stress_no_anchor["candidate_drives"]["action::replay_episode"],
        "stress_mismatch_boosts_replace": stress["candidate_drives"]["action::text_replace"] > stress_no_mismatch["candidate_drives"]["action::text_replace"],
        "stress_commit_ready_lower_than_baseline": stress["commit_readiness"] < baseline["commit_readiness"],
        "stress_selected_actions_include_revision_or_replay": any(
            aid in {"action::text_reread", "action::text_replace", "action::replay_episode"}
            for aid in stress["selected_actions"]
        ),
        "baseline_selected_actions_include_commit_or_insert": any(
            aid in {"action::text_commit", "action::text_insert"}
            for aid in baseline["selected_actions"]
        ),
    }
    passed = all(checks.values())
    return {
        "experiment": "DoubleEnergyBalance-PressureDynamics-1",
        "design": "Higher cognitive pressure should shift action competition toward reread / replace / replay and away from direct commit; removing pressure anchors or mismatch evidence should weaken the effect.",
        "verdict": _pass(passed, passed or checks["stress_reread_or_replace_outruns_commit"]),
        "observed": {
            "cases": cases,
            "checks": checks,
        },
        "boundary": {
            "ap_core_scope_only": True,
            "gl_learning_protocol_scope": False,
            "answer_table_lookup": False,
            "regex_route": False,
            "student_side_llm": False,
            "hidden_solver": False,
            "full_sentence_macro": False,
            "runtime_mechanism_modified": False,
            "ap_core_full_proof_claimed": False,
        },
    }


def run_double_energy_balance_pressure_sweep() -> dict:
    clean_cases = [
        _run_pressure_dynamics_case(name="clean_low", pressure=0.04, gap=0.0, anchor=0.0, include_mismatch=False),
        _run_pressure_dynamics_case(name="clean_mid", pressure=0.28, gap=0.0, anchor=0.0, include_mismatch=False),
        _run_pressure_dynamics_case(name="clean_high", pressure=0.52, gap=0.0, anchor=0.0, include_mismatch=False),
        _run_pressure_dynamics_case(name="clean_peak", pressure=0.78, gap=0.0, anchor=0.0, include_mismatch=False),
    ]
    stress_cases = [
        _run_pressure_dynamics_case(name="stress_low", pressure=0.04, gap=0.0, anchor=0.82, include_mismatch=True),
        _run_pressure_dynamics_case(name="stress_mid", pressure=0.28, gap=0.0, anchor=0.82, include_mismatch=True),
        _run_pressure_dynamics_case(name="stress_high", pressure=0.52, gap=0.0, anchor=0.82, include_mismatch=True),
        _run_pressure_dynamics_case(name="stress_peak", pressure=0.78, gap=0.0, anchor=0.82, include_mismatch=True),
    ]
    clean_commit_series = [row["candidate_drives"]["action::text_commit"] for row in clean_cases]
    clean_replay_series = [row["candidate_drives"]["action::replay_episode"] for row in clean_cases]
    stress_commit_series = [row["candidate_drives"]["action::text_commit"] for row in stress_cases]
    stress_replay_series = [row["candidate_drives"]["action::replay_episode"] for row in stress_cases]
    stress_revision_series = [
        max(row["candidate_drives"]["action::text_reread"], row["candidate_drives"]["action::text_replace"], row["candidate_drives"]["action::replay_episode"])
        for row in stress_cases
    ]
    checks = {
        "clean_commit_nonincreasing": all(clean_commit_series[idx] >= clean_commit_series[idx + 1] - 1e-6 for idx in range(len(clean_commit_series) - 1)),
        "clean_commit_remains_positive": clean_commit_series[-1] > 0.0 and clean_commit_series[0] > clean_commit_series[-1],
        "clean_replay_non_decreasing": all(clean_replay_series[idx] <= clean_replay_series[idx + 1] + 1e-6 for idx in range(len(clean_replay_series) - 1)),
        "stress_commit_suppressed": all(series <= 0.01 for series in stress_commit_series),
        "stress_replay_non_decreasing": all(stress_replay_series[idx] <= stress_replay_series[idx + 1] + 1e-6 for idx in range(len(stress_replay_series) - 1)),
        "stress_replay_outruns_clean_peak_replay": stress_replay_series[-1] > clean_replay_series[-1],
        "stress_peak_revision_outruns_clean_peak": stress_revision_series[-1] > max(
            clean_cases[-1]["candidate_drives"]["action::text_reread"],
            clean_cases[-1]["candidate_drives"]["action::text_replace"],
            clean_cases[-1]["candidate_drives"]["action::replay_episode"],
        ),
    }
    passed = all(checks.values())
    return {
        "experiment": "DoubleEnergyBalance-PressureDynamics-Sweep-1",
        "verdict": _pass(passed, passed or checks["stress_peak_revision_outruns_clean_peak"]),
        "design": "Two pressure sweeps should show the same underlying shape in both clean and stressed regimes: clean drafts stay commit-dominant while commit weakens with pressure, and stressed drafts shift toward replay / revision as pressure rises.",
        "observed": {
            "clean_cases": clean_cases,
            "stress_cases": stress_cases,
            "checks": checks,
            "clean_commit_series": clean_commit_series,
            "clean_replay_series": clean_replay_series,
            "stress_commit_series": stress_commit_series,
            "stress_replay_series": stress_replay_series,
            "stress_revision_series": stress_revision_series,
        },
        "boundary": {
            "ap_core_scope_only": True,
            "gl_learning_protocol_scope": False,
            "answer_table_lookup": False,
            "regex_route": False,
            "student_side_llm": False,
            "hidden_solver": False,
            "full_sentence_macro": False,
            "runtime_mechanism_modified": False,
            "ap_core_full_proof_claimed": False,
        },
    }


def _boundary() -> dict:
    return {
        "ap_core_scope_only": True,
        "gl_learning_protocol_scope": False,
        "answer_table_lookup": False,
        "regex_route": False,
        "student_side_llm": False,
        "hidden_solver": False,
        "full_sentence_macro": False,
    }


def run_apcore_dynamics_suite() -> dict:
    experiments = [
        run_feedback_override(),
        run_persistence_reload(),
        run_negative_feedback_ablation(),
        run_short_term_interruption_recovery(),
        run_residual_depth(),
        run_successor_peak_ablation(),
        run_double_energy_balance_pressure_dynamics(),
        run_double_energy_balance_pressure_sweep(),
    ]
    counts = {
        "pass": sum(1 for row in experiments if row.get("verdict") == "pass"),
        "partial": sum(1 for row in experiments if row.get("verdict") == "partial"),
        "fail": sum(1 for row in experiments if row.get("verdict") == "fail"),
    }
    return {
        "schema_id": "apv22_apcore_dynamics_suite/v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "route_split": {
            "this_suite": "AP-Core bottom-loop dynamics",
            "not_this_suite": "GL curriculum learning or skill-package generalization",
        },
        "experiments": experiments,
        "summary": {
            **counts,
            "all_passed": counts["fail"] == 0 and counts["partial"] == 0,
            "acceptance_policy": "pass_requires_mechanism_trace_and_module_specific_ablation_where_applicable",
        },
    }


def write_outputs(result: dict, *, output_root: Path | None = None) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root or (ROOT / "outputs" / f"apv22_apcore_dynamics_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "apv22_apcore_dynamics.json"
    md_path = out_dir / "apv22_apcore_dynamics_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# APV2.2 AP-Core Dynamics Report",
        "",
        f"- generated_at: `{result.get('generated_at')}`",
        "- scope: `AP-Core bottom-loop dynamics`",
        "- not_scope: `GL curriculum learning / skill-package generalization`",
        f"- all_passed: `{result.get('summary', {}).get('all_passed')}`",
        f"- pass: `{result.get('summary', {}).get('pass')}`",
        f"- partial: `{result.get('summary', {}).get('partial')}`",
        f"- fail: `{result.get('summary', {}).get('fail')}`",
        "",
        "## Experiments",
        "",
    ]
    for row in result.get("experiments", []) or []:
        lines.extend(
            [
                f"### {row.get('experiment')}",
                "",
                f"- verdict: `{row.get('verdict')}`",
                f"- design: {row.get('design')}",
                f"- boundary: `{json.dumps(row.get('boundary', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


def main() -> int:
    result = run_apcore_dynamics_suite()
    artifacts = write_outputs(result)
    print(json.dumps({"summary": result["summary"], "artifacts": artifacts}, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
