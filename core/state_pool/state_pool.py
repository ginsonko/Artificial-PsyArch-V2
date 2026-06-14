from __future__ import annotations

from dataclasses import dataclass, field
from heapq import nsmallest
from math import exp

from core.state_pool.energy_view import build_energy_flow_trace
from core.state_pool.energy_updater import PredictionEnergyUpdater
from core.state_pool.residual_tracker import ResidualTracker

"""
PHASE1_MINIMAL:
The dual-energy fields are already first-class here, but the update rules are
still deliberately simple. This module will later be split into more formal
subcomponents for entry storage, energy update, attention view, feeling
projection, and prediction tracing.

UPGRADE NOTE (2026-05-26):
二期草案硬约束要求：tick 热路径禁止全池遍历。状态池可以很大，但每 tick 参与召回与注意力的
查询体必须通过 `R_state` 固定预算、多头读出器产生。

本模块因此开始从“每 tick 全池衰减/排序”迁移到：
1) 懒惰衰减：只在对象被触达/读出时计算衰减
2) 分层旁路缓存：recent_external / hot_anchor / residual 等有界结构
3) R_state 多头固定预算读出：head_global/head_recent/head_anchor/head_prediction/head_residual

注意：`snapshot()` 仍是白箱 view（有界），不是状态池全量，也不是 R_state。
"""


def _round4(value: float) -> float:
    return round(float(value), 4)


@dataclass
class PoolEntry:
    sa_label: str
    display_text: str
    family: str
    source_type: str
    real_energy: float = 0.0
    virtual_energy: float = 0.0
    cognitive_pressure: float = 0.0
    attention_gain: float = 0.0
    fatigue: float = 0.0
    last_seen_tick: int = -1
    last_updated_tick: int = -1
    provenance: list[str] = field(default_factory=list)
    anchor_meta: dict = field(default_factory=dict)
    numeric_features: dict = field(default_factory=dict)
    reconstruction_payload: dict = field(default_factory=dict)

    def refresh_pressure(self) -> None:
        self.cognitive_pressure = float(self.real_energy) - float(self.virtual_energy)

    def as_dict(self, *, include_meta: bool = True) -> dict:
        self.refresh_pressure()
        row = {
            "sa_label": self.sa_label,
            "display_text": self.display_text,
            "family": self.family,
            "source_type": self.source_type,
            "real_energy": self.real_energy,
            "virtual_energy": self.virtual_energy,
            "cognitive_pressure": self.cognitive_pressure,
            "attention_gain": self.attention_gain,
            "fatigue": self.fatigue,
            "last_seen_tick": self.last_seen_tick,
            "last_updated_tick": self.last_updated_tick,
        }
        if include_meta:
            row["provenance"] = list(self.provenance)
            row["anchor_meta"] = dict(self.anchor_meta)
            if self.numeric_features:
                row["numeric_features"] = {
                    str(channel): list(values)
                    for channel, values in dict(self.numeric_features).items()
                }
            if self.reconstruction_payload:
                row["reconstruction_payload"] = dict(self.reconstruction_payload)
        else:
            row["is_focus"] = bool((self.anchor_meta or {}).get("is_focus", False))
            if "position" in self.anchor_meta:
                row["position"] = self.anchor_meta.get("position")
            if "tick_index" in self.anchor_meta:
                row["tick_index"] = self.anchor_meta.get("tick_index")
            if self.numeric_features:
                row["numeric_features"] = {
                    str(channel): list(values)
                    for channel, values in dict(self.numeric_features).items()
                }
            if self.reconstruction_payload:
                row["reconstruction_payload"] = dict(self.reconstruction_payload)
        return row


class DualEnergyStatePool:
    def __init__(
        self,
        *,
        real_decay: float,
        virtual_decay: float,
        attention_gain_decay: float,
        fatigue_decay: float,
        prune_threshold: float,
        query_limit: int,
        snapshot_limit: int,
        memory_snapshot_limit: int = 1024,
        r_state_head_limit: int = 5,
        r_state_items_per_head: int = 24,
        maintenance_budget: int = 48,
        recent_external_limit: int = 128,
        hot_anchor_limit: int = 256,
        prediction_validation_actual_limit: int = 256,
        prediction_validation_update_limit: int = 128,
        focus_boost: float,
        focus_fatigue_step: float,
        prediction_fatigue_enabled: bool = True,
        prediction_fatigue_min_mass: float = 0.18,
        prediction_fatigue_ratio: float = 0.18,
        prediction_fatigue_gain: float = 0.06,
        prediction_fatigue_max_step: float = 0.18,
        cstar_trace_top_labels: int = 8,
        bootstrap_virtual_energy: float,
    ) -> None:
        self.real_decay = float(real_decay)
        self.virtual_decay = float(virtual_decay)
        self.attention_gain_decay = float(attention_gain_decay)
        self.fatigue_decay = float(fatigue_decay)
        self.prune_threshold = float(prune_threshold)
        self.query_limit = int(query_limit)
        self.snapshot_limit = int(snapshot_limit)
        self.memory_snapshot_limit = max(int(snapshot_limit), int(memory_snapshot_limit))
        self.r_state_head_limit = max(1, int(r_state_head_limit))
        self.r_state_items_per_head = max(1, int(r_state_items_per_head))
        self.maintenance_budget = max(0, int(maintenance_budget))
        self.recent_external_limit = max(8, int(recent_external_limit))
        self.hot_anchor_limit = max(8, int(hot_anchor_limit))
        self.prediction_validation_actual_limit = max(16, int(prediction_validation_actual_limit))
        self.prediction_validation_update_limit = max(8, int(prediction_validation_update_limit))
        self.focus_boost = float(focus_boost)
        self.focus_fatigue_step = float(focus_fatigue_step)
        self.prediction_fatigue_enabled = bool(prediction_fatigue_enabled)
        self.prediction_fatigue_min_mass = max(0.0, float(prediction_fatigue_min_mass))
        self.prediction_fatigue_ratio = max(0.0, float(prediction_fatigue_ratio))
        self.prediction_fatigue_gain = max(0.0, float(prediction_fatigue_gain))
        self.prediction_fatigue_max_step = max(0.0, float(prediction_fatigue_max_step))
        self.cstar_trace_top_labels = max(1, int(cstar_trace_top_labels))
        self.bootstrap_virtual_energy = float(bootstrap_virtual_energy)
        self._entries: dict[str, PoolEntry] = {}
        self._entry_order: list[str] = []
        self._tick_index = -1
        # Fixed-budget caches (bounded): do not grow with pool size.
        self._recent_external: list[str] = []
        self._current_external_labels: list[str] = []
        self._current_source_types_by_label: dict[str, set[str]] = {}
        self._hot_anchor: list[str] = []
        self._hot_anchor_members: set[str] = set()
        self._residual_tracker = ResidualTracker(
            limit=max(64, self.hot_anchor_limit // 2),
            unit_limit_per_tick=max(8, self.r_state_items_per_head),
            decay=max(0.0, min(0.98, min(self.real_decay, self.virtual_decay))),
            prune_threshold=self.prune_threshold,
        )
        self._energy_updater = PredictionEnergyUpdater()
        self._pending_prediction_items: list[dict] = []
        self._current_focus_labels: set[str] = set()
        self._last_prediction_trace: dict = {
            "schema_id": "prediction_energy_trace/v1",
            "tick_index": -1,
            "predicted_labels": [],
            "actual_labels": [],
            "matched_labels": [],
            "missed_predicted_labels": [],
            "unexpected_labels": [],
            "predicted_mass": 0.0,
            "actual_mass": 0.0,
            "match_mass": 0.0,
            "mismatch_mass": 0.0,
            "alignment_score": 0.0,
            "mismatch_ratio": 0.0,
        }
        self._last_cstar_budget_trace: dict = {
            "schema_id": "cstar_budget_trace/v1",
            "tick_index": -1,
            "source": "",
            "energy_semantics": "same_label_sum_means_prediction_strength_not_occurrence_count",
            "total_virtual_mass": 0.0,
            "label_count": 0,
            "top_labels": [],
            "fatigue_updates": [],
            "budget_warnings": [],
        }
        self._prediction_slot: list[str] = []
        self._maintenance_cursor: int = 0
        self._current_external_real_baseline = 1.0
        self._last_external_real_baseline = 1.0
        self._tick_prediction_mass_by_label: dict[str, float] = {}
        self._tick_prediction_base_virtual_by_label: dict[str, float] = {}

    def begin_tick(self, tick_index: int) -> None:
        # DO NOT scan the full pool here.
        self._tick_index = int(tick_index)
        self._current_external_labels = []
        self._current_source_types_by_label = {}
        self._current_external_real_baseline = max(0.05, float(self._last_external_real_baseline or 1.0))
        self._tick_prediction_mass_by_label = {}
        self._tick_prediction_base_virtual_by_label = {}
        self._residual_tracker.begin_tick(self._tick_index)
        self._maintenance_step()

    def mark_external_turn_boundary(self, *, tick_index: int, reason: str = "new_external_text_turn") -> dict:
        """
        Let the previous turn's working-memory surface settle before a new user
        turn supplies fresh evidence.

        This is not forgetting and not a reply rule. It only applies bounded
        fatigue/settling to the hot working set so completed action feedback,
        old text surface, and process feelings do not keep masquerading as the
        current high-pressure field.
        """

        self._tick_index = int(tick_index)
        labels: list[str] = []
        labels.extend(reversed(self._recent_external[-self.recent_external_limit :]))
        labels.extend(reversed(self._hot_anchor[-self.hot_anchor_limit :]))
        labels.extend(reversed(self._prediction_slot[-max(8, self.r_state_items_per_head * 2) :]))
        labels.extend(sorted(self._current_focus_labels))
        try:
            for row in self._residual_tracker.items(limit=max(8, self.r_state_items_per_head)):
                label = str((row or {}).get("sa_label", "") or "")
                if label:
                    labels.append(label)
        except Exception:
            pass

        settle_families = {
            "action",
            "action_feedback",
            "action_inhibition",
            "action_control",
            "task",
            "intention",
            "cognitive_feeling",
            "expectation_pressure",
            "time_feeling",
            "rhythm_feeling",
            "short_term_slot",
            "text",
            "text_input",
            "text_action",
            "text_revision_opportunity",
            "process_feeling",
        }
        settle_sources = {
            "action_selection",
            "action_feedback",
            "action_control",
            "task_anchor",
            "intention_anchor",
            "dialogue_turn_state",
            "predicted",
            "internal_draft_visible",
            "internal_draft_read",
            "text_action",
            "short_term_slot",
            "short_term_echo",
            "thought_echo",
            "cognitive_feeling",
            "expectation_pressure",
            "time_feeling",
            "rhythm_feeling",
        }
        real_scale = max(0.18, float(self.real_decay) ** 8)
        virtual_scale = max(0.12, float(self.virtual_decay) ** 8)
        attention_scale = max(0.12, float(self.attention_gain_decay) ** 10)
        fatigue_step = max(0.0, float(self.focus_fatigue_step))
        touched = 0
        softened: list[dict] = []
        seen: set[str] = set()
        recent_external_before = len(self._recent_external)
        for raw_label in labels:
            label = str(raw_label or "")
            if not label or label in seen:
                continue
            seen.add(label)
            entry = self._entries.get(label)
            if entry is None:
                continue
            self._touch_entry(entry)
            family = str(entry.family or "")
            source_type = str(entry.source_type or "")
            if family not in settle_families and source_type not in settle_sources:
                continue
            before = {
                "real": _round4(entry.real_energy),
                "virtual": _round4(entry.virtual_energy),
                "attention": _round4(entry.attention_gain),
                "fatigue": _round4(entry.fatigue),
            }
            entry.real_energy = float(entry.real_energy) * real_scale
            entry.virtual_energy = float(entry.virtual_energy) * virtual_scale
            entry.attention_gain = float(entry.attention_gain) * attention_scale
            entry.fatigue = float(entry.fatigue) + fatigue_step
            entry.last_updated_tick = int(tick_index)
            entry.anchor_meta["last_external_turn_boundary_reason"] = str(reason or "")
            entry.provenance.append(f"external_turn_boundary@{tick_index}")
            entry.provenance = entry.provenance[-12:]
            entry.refresh_pressure()
            touched += 1
            if len(softened) < 12:
                softened.append(
                    {
                        "sa_label": label,
                        "family": family,
                        "source_type": source_type,
                        "before": before,
                        "after": {
                            "real": _round4(entry.real_energy),
                            "virtual": _round4(entry.virtual_energy),
                            "attention": _round4(entry.attention_gain),
                            "fatigue": _round4(entry.fatigue),
                        },
                    }
                )
        # A new external turn opens a fresh sensory/text surface. Previous
        # external labels have already been settled above and remain available
        # through short-term/long-term memory; keeping them in the same
        # recent-external queue makes old text masquerade as current input.
        if recent_external_before:
            self._recent_external = []
        return {
            "schema_id": "state_pool_external_turn_boundary/v1",
            "applied": bool(touched > 0),
            "tick_index": int(tick_index),
            "reason": str(reason or ""),
            "softened_count": int(touched),
            "recent_external_cleared_count": int(recent_external_before),
            "softened_preview": softened,
            "policy": "bounded_working_memory_settling_not_long_term_forgetting",
        }

    def _maintenance_step(self) -> None:
        """
        Fixed-budget maintenance step.

        Why it exists:
        - With lazy decay, entries that are never touched could otherwise live forever.
        - 二期草案允许用固定最大增量/维护预算把成本钉死。
        """

        budget = int(self.maintenance_budget)
        if budget <= 0:
            return
        if not self._entries:
            self._maintenance_cursor = 0
            return
        keys = self._entry_order
        if not keys:
            self._maintenance_cursor = 0
            return
        start = int(self._maintenance_cursor) % len(keys)
        checked = 0
        idx = start
        stale: list[str] = []
        while checked < budget and keys:
            label = keys[idx % len(keys)]
            entry = self._entries.get(label)
            if entry is not None:
                self._touch_entry(entry)
                if max(entry.real_energy, entry.virtual_energy) < self.prune_threshold:
                    stale.append(label)
            checked += 1
            idx += 1
        for label in stale:
            self._entries.pop(label, None)
            self._remove_from_caches(label)
        self._maintenance_cursor = idx % max(1, len(keys))

    def _remove_from_caches(self, label: str) -> None:
        clean = str(label or "")
        if not clean:
            return
        if clean in self._recent_external:
            self._recent_external = [x for x in self._recent_external if x != clean]
        if clean in self._hot_anchor:
            self._hot_anchor = [x for x in self._hot_anchor if x != clean]
            self._hot_anchor_members.discard(clean)
        if clean in self._prediction_slot:
            self._prediction_slot = [x for x in self._prediction_slot if x != clean]
        if clean in self._entry_order:
            self._entry_order = [x for x in self._entry_order if x != clean]
        self._residual_tracker.remove(clean)

    def _touch_entry(self, entry: PoolEntry) -> None:
        """
        Apply lazy decay from `entry.last_updated_tick` to current tick.
        """

        current = int(self._tick_index)
        last = int(entry.last_updated_tick)
        if last < 0:
            entry.last_updated_tick = current
            entry.refresh_pressure()
            return
        steps = max(0, current - last)
        if steps <= 0:
            entry.refresh_pressure()
            return
        # Exponential decay across multiple ticks.
        entry.real_energy = float(entry.real_energy) * (self.real_decay ** steps)
        entry.virtual_energy = float(entry.virtual_energy) * (self.virtual_decay ** steps)
        entry.attention_gain = float(entry.attention_gain) * (self.attention_gain_decay ** steps)
        entry.fatigue = float(entry.fatigue) * (self.fatigue_decay ** steps)
        entry.last_updated_tick = current
        entry.refresh_pressure()

    def _is_external_source(self, source_type: str) -> bool:
        src = str(source_type or "")
        if src == "external_text":
            return True
        if src == "external_teacher" or src.endswith("_external_teacher"):
            return True
        if src.startswith("vision_bridge"):
            return True
        if src.startswith("audio_bridge"):
            return True
        if src == "vision_numeric":
            return True
        if src == "audio_numeric":
            return True
        return False

    def _is_refresh_observation(self, item: dict, *, incoming_source_type: str) -> bool:
        """Return True when an incoming SA is a current-field refresh.

        These sources describe what the runtime is currently feeling, doing,
        or seeing in its own draft/action surface. Their learning value is
        written through memory/action-feedback paths; the state-pool amplitude
        itself should not grow just because the same process is refreshed every
        tick.
        """

        src = str(incoming_source_type or item.get("source_type", "") or "")
        family = str(item.get("family", "") or "")
        label = str(item.get("sa_label", "") or "")
        if src in {
            "dialogue_turn_state",
            "task_anchor",
            "intention_anchor",
            "action_selection",
            "action_feedback",
            "action_control",
            "text_action",
            "internal_draft_visible",
            "internal_draft_read",
            "short_term_slot",
            "short_term_echo",
            "thought_echo",
            "cognitive_feeling",
            "expectation_pressure",
            "time_feeling",
            "rhythm_feeling",
            "runtime_load_feeling",
            "runtime_load",
            "safety_gate",
            "innate_rule",
        }:
            return True
        if family in {
            "task",
            "intention",
            "action",
            "action_feedback",
            "action_inhibition",
            "action_control",
            "text_action",
            "text_revision_opportunity",
            "short_term_slot",
            "short_term_echo",
            "cognitive_feeling",
            "expectation_pressure",
            "time_feeling",
            "rhythm_feeling",
            "runtime_load_feeling",
        }:
            return True
        return label.startswith(("action::", "action_feedback::", "text_action::", "text_revision_opportunity::"))

    def _has_process_anchor_meta(self, meta: dict) -> bool:
        """
        Preserve metadata only for low-grain process anchors.

        Ordinary `text::x` identities stay compact in memory snapshots. V2
        teaching rows, however, use metadata to bind a character to cursor,
        slot, readout-frame, feedback, and process-feeling roles. Dropping that
        metadata turns the learned candidate back into naked text familiarity.
        These fields describe the learning process, not a full answer table.
        """

        if not isinstance(meta, dict) or not meta:
            return False
        schema_id = str(meta.get("schema_id", "") or "")
        if schema_id in {
            "text_revision_opportunity/v1",
            "text_slot_confirmation/v1",
            "text_character_binding/v1",
            "text_action_feedback/v1",
            "text_insert_closure_state/v1",
            "text_reread_closure_state/v1",
            "text_cursor_state/v1",
            "ui_readout3_whole_region_time_readout_frame/v1",
        }:
            return True
        if str(meta.get("process_anchor_role", "") or ""):
            return True
        if str(meta.get("readout_pattern_id", "") or "") or str(meta.get("readout_semantic_role", "") or ""):
            return True
        if str(meta.get("prediction_payload_priority", "") or "").startswith(("current_glyph", "previous_prefix")):
            return True
        if bool(meta.get("current_read_tick", False)):
            return True
        if meta.get("current_glyph_index") is not None and (
            str(meta.get("current_glyph_role", "") or "")
            or str(meta.get("same_tick_binding_role", "") or "")
            or str(meta.get("slot_role", "") or "")
        ):
            return True
        return False

    def _process_anchor_merge_keys(self) -> set[str]:
        return {
            "schema_id",
            "event_type",
            "process_anchor_role",
            "readout_pattern_id",
            "readout_semantic_role",
            "semantic_frame_role",
            "dynamic_slot_role",
            "slot_role",
            "current_glyph_index",
            "visible_length",
            "cursor",
            "cursor_index",
            "current_glyph_role",
            "same_tick_binding_role",
            "prediction_payload_priority",
            "current_read_tick",
            "previous_prefix",
            "last_visible_token",
            "teacher_label_is_scaffold",
            "teacher_on_character_label",
            "teacher_on_only",
            "used_in_strict_teacher_off_input",
            "answer_table_lookup",
            "full_string_or_sentence_action",
            "external_surface_hard_gate",
        }

    def _process_anchor_meta_score(self, meta: dict) -> int:
        if not isinstance(meta, dict) or not meta:
            return 0
        score = 0
        for key in self._process_anchor_merge_keys():
            value = meta.get(key)
            if value is not None and str(value) != "":
                score += 1
        if self._has_process_anchor_meta(meta):
            score += 8
        return score

    def _merge_anchor_meta(self, existing: dict, incoming: dict) -> dict:
        """
        Merge metadata without erasing learned V2 process anchors.

        The state pool stores one entry per SA label. For UI readout learning,
        the same low-grain `text::0` may be touched many times. A later generic
        touch should not strip the slot/readout-frame metadata that makes the
        memory useful, while a later richer process touch should still update
        the current slot. This is retention of already-observed process
        metadata, not answer injection.
        """

        old = dict(existing or {}) if isinstance(existing, dict) else {}
        new = dict(incoming or {}) if isinstance(incoming, dict) else {}
        if not new:
            return old
        if not old:
            return new

        old_process_score = self._process_anchor_meta_score(old)
        new_process_score = self._process_anchor_meta_score(new)
        process_keys = self._process_anchor_merge_keys()
        merged = dict(old)
        for key, value in new.items():
            if key in process_keys and old_process_score > 0 and new_process_score <= 0:
                continue
            if key in process_keys and (value is None or str(value) == "") and old.get(key) not in (None, ""):
                continue
            merged[key] = value
        return merged

    def _is_prediction_comparable_label(self, label: str) -> bool:
        clean = str(label or "")
        if not clean:
            return False
        if clean.startswith(("text::", "phrase::", "audio::", "audio_event::", "vision::", "vision_obj::", "vision_mem::")):
            return True
        return False

    def _is_prediction_comparable_item(self, item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        label = str(item.get("sa_label", "") or "")
        if self._is_prediction_comparable_label(label):
            return True
        family = str(item.get("family", "") or "")
        if family in {"text", "learned_text_phrase", "vision", "vision_channel", "vision_object", "audio", "audio_channel", "audio_event"}:
            return True
        return False

    def apply_external_items(self, items: list[dict], *, tick_index: int) -> None:
        comparable_actual_items = []
        external_real_samples: list[float] = []
        if items and (self._pending_prediction_items or len(items) <= self.prediction_validation_actual_limit):
            for item in items:
                if len(comparable_actual_items) >= self.prediction_validation_actual_limit:
                    break
                if not isinstance(item, dict) or not self._is_prediction_comparable_item(item):
                    continue
                comparable_actual_items.append(
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "display_text": str(item.get("display_text", item.get("sa_label", "")) or item.get("sa_label", "")),
                        "family": str(item.get("family", "") or ""),
                        "source_type": str(item.get("source_type", "") or ""),
                        "real_energy": float(item.get("real_energy", 1.0) or 0.0),
                        "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                    }
                )
        recent_external_dirty = False
        hot_anchor_dirty = False
        for item in items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            incoming_source_type = str(item.get("source_type", "") or "")
            if not incoming_source_type:
                incoming_source_type = "external"
            self._current_source_types_by_label.setdefault(label, set()).add(incoming_source_type)
            entry = self._entries.get(label)
            if entry is None:
                entry = PoolEntry(
                    sa_label=label,
                    display_text=str(item.get("display_text", label) or label),
                    family=str(item.get("family", "text") or "text"),
                    source_type=incoming_source_type,
                )
                entry.last_updated_tick = int(tick_index)
                self._entries[label] = entry
                self._entry_order.append(label)
            self._touch_entry(entry)
            # Do not overwrite stable identity metadata (family / display_text / source_type)
            # for an already-existing SA object. Control injections and other non-sensory
            # updates should modulate energies, not rewrite what the SA "is".
            # If a later stage wants different display, it should use `anchor_meta`.
            is_action_control_target = incoming_source_type == "action_control" and not label.startswith("control::")
            if is_action_control_target:
                # Action-control target rows are attention hints for existing
                # SA objects, not evidence that the concept itself gained real
                # or virtual predictive energy. This preserves the AP boundary:
                # action feedback/control can shape future actions, but should
                # not teach the concept embedding space.
                entry.attention_gain = float(entry.attention_gain) + max(
                    0.0,
                    float(item.get("attention_gain", item.get("virtual_energy", 0.0)) or 0.0),
                )
            elif self._is_refresh_observation(item, incoming_source_type=incoming_source_type):
                # Momentary process observations are amplitudes of the current
                # field, not evidence-count increments. Re-applying
                # dialogue-turn state, action feedback, text-action readback,
                # or cognitive feelings every tick must refresh the visible
                # process surface instead of making old internal residue grow
                # without bound and outcompete fresh external evidence.
                entry.real_energy = max(0.0, float(item.get("real_energy", 1.0) or 0.0))
                entry.virtual_energy = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
                entry.attention_gain = max(0.0, float(item.get("attention_gain", 0.0) or 0.0))
            else:
                entry.real_energy = float(entry.real_energy) + float(item.get("real_energy", 1.0) or 0.0)
                entry.virtual_energy = float(entry.virtual_energy) + float(item.get("virtual_energy", 0.0) or 0.0)
            if isinstance(item.get("anchor_meta"), dict):
                entry.anchor_meta = self._merge_anchor_meta(
                    entry.anchor_meta,
                    dict(item.get("anchor_meta", {}) or {}),
                )
            if isinstance(item.get("numeric_features"), dict):
                entry.numeric_features = {
                    str(channel): list(values if isinstance(values, (list, tuple)) else [values])
                    for channel, values in dict(item.get("numeric_features", {}) or {}).items()
                    if str(channel or "")
                }
            if isinstance(item.get("reconstruction_payload"), dict):
                entry.reconstruction_payload = dict(item.get("reconstruction_payload", {}) or {})
            if "position" in item:
                entry.anchor_meta["position"] = item.get("position")
            entry.anchor_meta["tick_index"] = int(tick_index)
            entry.last_seen_tick = int(tick_index)
            entry.provenance.append(f"{incoming_source_type}@{tick_index}")
            entry.provenance = entry.provenance[-12:]
            entry.last_updated_tick = int(tick_index)
            entry.refresh_pressure()
            # Maintain bounded caches.
            if self._is_external_source(incoming_source_type):
                self._recent_external.append(label)
                self._current_external_labels.append(label)
                external_real_samples.append(max(0.0, float(entry.real_energy or 0.0)))
                recent_external_dirty = True
            # Hot anchor: keep high-salience entries quickly addressable without full sorts.
            # We bias toward items with positive real energy and high attention_gain.
            salience = float(entry.real_energy + entry.attention_gain - entry.fatigue * 0.2)
            if salience > 0.2 and label not in self._hot_anchor_members:
                self._hot_anchor.append(label)
                self._hot_anchor_members.add(label)
                hot_anchor_dirty = True
            elif salience > 0.2 and label in self._hot_anchor_members:
                # Keep hot-anchor updates O(1). Recency-sensitive evidence is
                # represented by recent_external; hot_anchor is a membership cache.
                pass
        if recent_external_dirty and len(self._recent_external) > self.recent_external_limit:
            self._recent_external = self._recent_external[-self.recent_external_limit :]
        if hot_anchor_dirty and len(self._hot_anchor) > self.hot_anchor_limit:
            removed_hot = self._hot_anchor[: len(self._hot_anchor) - self.hot_anchor_limit]
            self._hot_anchor = self._hot_anchor[-self.hot_anchor_limit :]
            for old_label in removed_hot:
                self._hot_anchor_members.discard(old_label)
        if external_real_samples:
            positive = [value for value in external_real_samples if value > 0.0]
            if positive:
                baseline = sum(positive) / max(1, len(positive))
                self._current_external_real_baseline = max(0.05, baseline)
                self._last_external_real_baseline = self._current_external_real_baseline
        if comparable_actual_items or self._pending_prediction_items:
            self.validate_predictions(comparable_actual_items, tick_index=tick_index)

    def apply_predictions(self, items: list[dict], *, tick_index: int, source: str) -> None:
        comparable_predictions: list[dict] = []
        cstar_trace = self._build_cstar_budget_trace(items, tick_index=tick_index, source=source)
        calibration_updates: list[dict] = []
        for item in items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            entry = self._entries.get(label)
            if entry is None:
                entry = PoolEntry(
                    sa_label=label,
                    display_text=str(item.get("display_text", label) or label),
                    family=str(item.get("family", "predicted") or "predicted"),
                    source_type=str(item.get("source_type", "predicted") or "predicted"),
                )
                entry.last_updated_tick = int(tick_index)
                self._entries[label] = entry
                self._entry_order.append(label)
            self._touch_entry(entry)
            entry.display_text = str(item.get("display_text", entry.display_text) or entry.display_text)
            incoming_virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            if label not in self._tick_prediction_base_virtual_by_label:
                self._tick_prediction_base_virtual_by_label[label] = max(0.0, float(entry.virtual_energy or 0.0))
            self._tick_prediction_mass_by_label[label] = (
                max(0.0, float(self._tick_prediction_mass_by_label.get(label, 0.0) or 0.0))
                + incoming_virtual
            )
            correction = self._calibrated_prediction_virtual_energy(
                entry=entry,
                item=item,
                label=label,
                incoming_virtual=incoming_virtual,
                tick_prediction_mass=self._tick_prediction_mass_by_label[label],
                source=source,
            )
            entry.virtual_energy = float(correction["after_virtual_energy"])
            if len(calibration_updates) < self.cstar_trace_top_labels:
                calibration_updates.append(correction)
            entry.anchor_meta["last_prediction_source"] = source
            entry.anchor_meta["last_prediction_energy_calibration"] = dict(correction)
            entry.provenance.append(f"{source}@{tick_index}")
            entry.provenance = entry.provenance[-12:]
            entry.last_updated_tick = int(tick_index)
            entry.refresh_pressure()
            # Prediction slot is a bounded head used by R_state; keep it small.
            if label not in self._prediction_slot:
                self._prediction_slot.append(label)
                if len(self._prediction_slot) > max(8, self.r_state_items_per_head * 2):
                    self._prediction_slot = self._prediction_slot[-max(8, self.r_state_items_per_head * 2) :]
            if self._is_prediction_comparable_item(item):
                comparable_predictions.append(
                    {
                        "sa_label": label,
                        "display_text": str(item.get("display_text", label) or label),
                        "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                    }
                )
        aggregate_trace = self._merge_cstar_budget_trace(cstar_trace)
        cstar_trace["state_pool_energy_calibration"] = calibration_updates[: self.cstar_trace_top_labels]
        aggregate_trace["state_pool_energy_calibration"] = (
            list(aggregate_trace.get("state_pool_energy_calibration", []) or [])
            + calibration_updates
        )[-self.cstar_trace_top_labels :]
        fatigue_updates = self._apply_prediction_fatigue(cstar_trace, tick_index=tick_index, source=source)
        cstar_trace["fatigue_updates"] = fatigue_updates
        aggregate_trace["fatigue_updates"] = (list(aggregate_trace.get("fatigue_updates", []) or []) + fatigue_updates)[-self.cstar_trace_top_labels :]
        self._last_cstar_budget_trace = aggregate_trace
        if comparable_predictions:
            self._pending_prediction_items = (self._pending_prediction_items + comparable_predictions)[-max(16, self.r_state_items_per_head * 4) :]

    def _prediction_real_baseline(self) -> float:
        """
        Return the current AP energy ruler.

        Fresh external evidence defines the real-energy scale for the tick. In
        empty ticks AP still keeps the last external scale as an internal
        calibration ruler, so repeated imagination cannot create a larger
        "reality" merely by being replayed.
        """

        return max(0.05, float(self._current_external_real_baseline or self._last_external_real_baseline or 1.0))

    def _prediction_support_signals(self, item: dict) -> dict:
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        transfer = dict(meta.get("prediction_energy_transfer", {}) or {}) if isinstance(meta.get("prediction_energy_transfer", {}), dict) else {}
        reward = max(0.0, float(meta.get("feedback_reward", item.get("feedback_reward", 0.0)) or 0.0))
        correctness = max(0.0, float(meta.get("feedback_correctness", item.get("feedback_correctness", 0.0)) or 0.0))
        punishment = max(0.0, float(meta.get("feedback_punishment", item.get("feedback_punishment", 0.0)) or 0.0))
        source_b_match = max(0.0, float(transfer.get("source_b_match_efficiency", 0.0) or 0.0))
        successor_weight = max(0.0, float(transfer.get("successor_weight", 0.0) or 0.0))
        calibration_gain = max(0.0, float(transfer.get("calibration_gain", 1.0) or 1.0))
        positive_outcome = max(0.0, reward + correctness * 0.35 - punishment)
        repeated_successor_support = max(0.0, source_b_match * successor_weight * min(2.0, calibration_gain))
        return {
            "reward": reward,
            "correctness": correctness,
            "punishment": punishment,
            "positive_outcome": positive_outcome,
            "source_b_match_efficiency": source_b_match,
            "successor_weight": successor_weight,
            "calibration_gain": calibration_gain,
            "repeated_successor_support": repeated_successor_support,
        }

    def _calibrated_prediction_virtual_energy(
        self,
        *,
        entry: PoolEntry,
        item: dict,
        label: str,
        incoming_virtual: float,
        tick_prediction_mass: float,
        source: str,
    ) -> dict:
        """
        Apply minimum-prediction-error calibration to Cn/C* state-pool writes.

        Cn still supplies the predicted item and the C* trace still records the
        raw prediction mass. The state-pool amplitude, however, is a bounded
        expectation under the current real-energy ruler. Repeating the same
        prediction across empty ticks therefore approaches a stable virtual
        target instead of accumulating into a hallucinated external stimulus.
        """

        before = max(0.0, float(entry.virtual_energy or 0.0))
        baseline = self._prediction_real_baseline()
        support = self._prediction_support_signals(item)
        base_ratio = max(0.05, min(1.0, float(self.bootstrap_virtual_energy or 0.6)))
        support_signal = max(
            0.0,
            float(support["repeated_successor_support"])
            + float(support["positive_outcome"]),
        )
        support_level = 1.0 - exp(-support_signal)
        punishment_level = 1.0 - exp(-max(0.0, float(support["punishment"])))
        ordinary_target = baseline * base_ratio
        supported_target = baseline * (base_ratio + (1.0 - base_ratio) * support_level)
        reward_lift = baseline * max(0.0, float(self.focus_boost or 0.0)) * (1.0 - exp(-float(support["positive_outcome"])))
        target_cap = max(0.0, supported_target + reward_lift - baseline * max(0.0, float(self.prediction_fatigue_gain or 0.0)) * punishment_level)
        target_cap = max(min(ordinary_target, baseline), target_cap)
        # C* can contain several same-label branches in one tick. Their raw
        # mass is visible in the trace; the state-pool target preserves it
        # while it stays below the current real-energy ruler, and only
        # saturates when repeated support would inflate imagination past that
        # ruler.
        target_signal = min(max(0.0, float(tick_prediction_mass)), target_cap)
        if before <= target_signal:
            after = target_signal
            correction_kind = "prediction_uptake_to_saturating_target"
        else:
            miss_decay = max(0.0, min(1.0, float(getattr(self._energy_updater, "miss_virtual_decay", 0.52) or 0.52)))
            after = target_signal + (before - target_signal) * miss_decay
            correction_kind = "minimum_prediction_error_overprediction_decay"
        return {
            "sa_label": str(label),
            "source": str(source or ""),
            "baseline_real_energy": _round4(baseline),
            "incoming_virtual_energy": _round4(incoming_virtual),
            "tick_prediction_mass": _round4(tick_prediction_mass),
            "ordinary_target": _round4(ordinary_target),
            "target_cap": _round4(target_cap),
            "target_signal": _round4(target_signal),
            "before_virtual_energy": _round4(before),
            "after_virtual_energy": _round4(after),
            "correction_kind": correction_kind,
            "support": {
                key: _round4(value) if isinstance(value, float) else value
                for key, value in support.items()
            },
            "policy": "raw_cstar_mass_is_audit;state_pool_virtual_energy_approaches_real_baseline_calibrated_prediction_target",
        }

    def _build_cstar_budget_trace(self, items: list[dict], *, tick_index: int, source: str) -> dict:
        """
        Build a bounded white-box audit for the C* merge performed by
        `apply_predictions`.

        This method deliberately does not change energy. Its job is to keep AP's
        prediction semantics visible: same-label virtual energy means prediction
        strength under the current budget, not repeated occurrence count.
        """

        by_label: dict[str, dict] = {}
        total_virtual_mass = 0.0
        budget_warnings: list[dict] = []
        input_count = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            input_count += 1
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            total_virtual_mass += virtual
            bucket = by_label.setdefault(
                label,
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "virtual_mass": 0.0,
                    "item_count": 0,
                    "source_branches": [],
                },
            )
            bucket["virtual_mass"] = float(bucket.get("virtual_mass", 0.0) or 0.0) + virtual
            bucket["item_count"] = int(bucket.get("item_count", 0) or 0) + 1
            meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            transfer = dict(meta.get("prediction_energy_transfer", {}) or {}) if isinstance(meta.get("prediction_energy_transfer", {}), dict) else {}
            if transfer:
                branch = {
                    "source_memory_id": str(transfer.get("source_memory_id", "") or ""),
                    "successor_memory_id": str(transfer.get("successor_memory_id", "") or ""),
                    "source_b_weight": _round4(float(transfer.get("source_b_weight", 0.0) or 0.0)),
                    "successor_weight": _round4(float(transfer.get("successor_weight", 0.0) or 0.0)),
                    "payload_share": _round4(float(transfer.get("payload_share", 0.0) or 0.0)),
                    "virtual_energy": _round4(virtual),
                }
                branches = list(bucket.get("source_branches", []) or [])
                if len(branches) < 4:
                    branches.append(branch)
                    bucket["source_branches"] = branches
                semantics = str(transfer.get("energy_budget_semantics", "") or "")
                if semantics and semantics != "virtual_energy_is_prediction_strength_not_occurrence_count":
                    budget_warnings.append(
                        {
                            "sa_label": label,
                            "warning": "unexpected_prediction_energy_semantics",
                            "value": semantics,
                        }
                    )
                calibrated = float(transfer.get("calibrated_transfer_multiplier", transfer.get("transfer_multiplier", 0.0)) or 0.0)
                raw = float(transfer.get("transfer_multiplier", calibrated) or 0.0)
                if calibrated > raw + 1e-6:
                    budget_warnings.append(
                        {
                            "sa_label": label,
                            "warning": "post_normalization_energy_gain_detected",
                            "transfer_multiplier": _round4(raw),
                            "calibrated_transfer_multiplier": _round4(calibrated),
                        }
                    )
        top_labels = sorted(
            by_label.values(),
            key=lambda row: (-float(row.get("virtual_mass", 0.0) or 0.0), str(row.get("sa_label", "") or "")),
        )[: self.cstar_trace_top_labels]
        for row in top_labels:
            total = max(1e-9, total_virtual_mass)
            row["virtual_mass"] = _round4(float(row.get("virtual_mass", 0.0) or 0.0))
            row["share_of_cstar"] = _round4(float(row.get("virtual_mass", 0.0) or 0.0) / total)
            row["item_count"] = int(row.get("item_count", 0) or 0)
        return {
            "schema_id": "cstar_budget_trace/v1",
            "tick_index": int(tick_index),
            "trace_scope": "prediction_branch",
            "source": str(source or ""),
            "energy_semantics": "same_label_sum_means_prediction_strength_not_occurrence_count",
            "policy": "raw_cstar_audit_state_pool_write_uses_min_prediction_error_calibration",
            "input_item_count": int(input_count),
            "total_virtual_mass": _round4(total_virtual_mass),
            "label_count": len(by_label),
            "top_labels": top_labels,
            "budget_warnings": budget_warnings[: self.cstar_trace_top_labels],
            "fatigue_updates": [],
        }

    def _merge_cstar_budget_trace(self, branch_trace: dict) -> dict:
        """
        Merge one prediction branch into the tick-level C* audit.

        Runtime can call `apply_predictions` more than once in a tick (fast
        recall, timefelt recall, slow recall). C* is the union of those branches,
        so the public audit keeps both bounded branch previews and a same-label
        tick aggregate.
        """

        tick = int(branch_trace.get("tick_index", self._tick_index))
        last_tick = int(self._last_cstar_budget_trace.get("tick_index", -1))
        if last_tick != tick:
            aggregate = {
                "schema_id": "cstar_budget_trace/v1",
                "tick_index": tick,
                "trace_scope": "tick_cstar",
                "source": "tick_aggregate",
                "energy_semantics": "same_label_sum_means_prediction_strength_not_occurrence_count",
                "policy": "raw_cstar_audit_state_pool_write_uses_min_prediction_error_calibration",
                "input_item_count": 0,
                "total_virtual_mass": 0.0,
                "label_count": 0,
                "top_labels": [],
                "branches": [],
                "budget_warnings": [],
                "fatigue_updates": [],
            }
        else:
            aggregate = dict(self._last_cstar_budget_trace)
            aggregate.setdefault("branches", [])
            aggregate.setdefault("budget_warnings", [])
            aggregate.setdefault("fatigue_updates", [])

        by_label: dict[str, dict] = {
            str(row.get("sa_label", "") or ""): dict(row)
            for row in list(aggregate.get("top_labels", []) or [])
            if str(row.get("sa_label", "") or "")
        }
        for row in list(branch_trace.get("top_labels", []) or []):
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            bucket = by_label.setdefault(
                label,
                {
                    "sa_label": label,
                    "display_text": str(row.get("display_text", label) or label),
                    "virtual_mass": 0.0,
                    "item_count": 0,
                    "source_branches": [],
                },
            )
            bucket["virtual_mass"] = float(bucket.get("virtual_mass", 0.0) or 0.0) + float(row.get("virtual_mass", 0.0) or 0.0)
            bucket["item_count"] = int(bucket.get("item_count", 0) or 0) + int(row.get("item_count", 0) or 0)
            source_branches = list(bucket.get("source_branches", []) or [])
            for branch in list(row.get("source_branches", []) or []):
                if len(source_branches) >= 4:
                    break
                source_branches.append(dict(branch))
            bucket["source_branches"] = source_branches

        total_virtual = float(aggregate.get("total_virtual_mass", 0.0) or 0.0) + float(branch_trace.get("total_virtual_mass", 0.0) or 0.0)
        top_labels = sorted(
            by_label.values(),
            key=lambda row: (-float(row.get("virtual_mass", 0.0) or 0.0), str(row.get("sa_label", "") or "")),
        )[: self.cstar_trace_top_labels]
        for row in top_labels:
            total = max(1e-9, total_virtual)
            row["virtual_mass"] = _round4(float(row.get("virtual_mass", 0.0) or 0.0))
            row["share_of_cstar"] = _round4(float(row.get("virtual_mass", 0.0) or 0.0) / total)
            row["item_count"] = int(row.get("item_count", 0) or 0)

        branches = list(aggregate.get("branches", []) or [])
        branches.append(
            {
                "source": str(branch_trace.get("source", "") or ""),
                "input_item_count": int(branch_trace.get("input_item_count", 0) or 0),
                "total_virtual_mass": _round4(float(branch_trace.get("total_virtual_mass", 0.0) or 0.0)),
                "label_count": int(branch_trace.get("label_count", 0) or 0),
                "top_labels": list(branch_trace.get("top_labels", []) or [])[: min(4, self.cstar_trace_top_labels)],
            }
        )
        aggregate["input_item_count"] = int(aggregate.get("input_item_count", 0) or 0) + int(branch_trace.get("input_item_count", 0) or 0)
        aggregate["total_virtual_mass"] = _round4(total_virtual)
        aggregate["label_count"] = len(by_label)
        aggregate["top_labels"] = top_labels
        aggregate["branches"] = branches[-self.cstar_trace_top_labels :]
        aggregate["budget_warnings"] = (list(aggregate.get("budget_warnings", []) or []) + list(branch_trace.get("budget_warnings", []) or []))[-self.cstar_trace_top_labels :]
        return aggregate

    def _apply_prediction_fatigue(self, cstar_trace: dict, *, tick_index: int, source: str) -> list[dict]:
        """
        Add small short-term fatigue to dominant predicted labels.

        This is not an energy correction. Virtual energy stays intact as a
        background expectation; fatigue only reduces the chance that the same
        label monopolizes the next attention/readout competition.
        """

        if not self.prediction_fatigue_enabled:
            return []
        total = max(0.0, float(cstar_trace.get("total_virtual_mass", 0.0) or 0.0))
        if total <= 0.0:
            return []
        threshold = max(self.prediction_fatigue_min_mass, total * self.prediction_fatigue_ratio)
        updates: list[dict] = []
        for row in list(cstar_trace.get("top_labels", []) or []):
            label = str(row.get("sa_label", "") or "")
            mass = max(0.0, float(row.get("virtual_mass", 0.0) or 0.0))
            if not label or mass < threshold:
                continue
            entry = self._entries.get(label)
            if entry is None:
                continue
            self._touch_entry(entry)
            before = float(entry.fatigue)
            share = mass / max(total, 1e-9)
            step = min(self.prediction_fatigue_max_step, self.prediction_fatigue_gain * (mass / (mass + threshold)) * (0.65 + share))
            if step <= 0.0:
                continue
            entry.fatigue = float(entry.fatigue) + step
            entry.anchor_meta["last_prediction_fatigue_source"] = source
            entry.provenance.append(f"prediction_fatigue@{tick_index}")
            entry.provenance = entry.provenance[-12:]
            entry.last_updated_tick = int(tick_index)
            entry.refresh_pressure()
            updates.append(
                {
                    "sa_label": label,
                    "virtual_mass": _round4(mass),
                    "share_of_cstar": _round4(share),
                    "threshold": _round4(threshold),
                    "fatigue_before": _round4(before),
                    "fatigue_added": _round4(step),
                    "fatigue_after": _round4(entry.fatigue),
                }
            )
        return updates[: self.cstar_trace_top_labels]

    def validate_predictions(self, actual_items: list[dict], *, tick_index: int) -> dict:
        predicted_items = [dict(item) for item in self._pending_prediction_items if isinstance(item, dict)]
        actual_rows = [dict(item) for item in (actual_items or []) if self._is_prediction_comparable_item(item)]
        if not predicted_items and not actual_rows:
            return dict(self._last_prediction_trace)
        trace = self._energy_updater.build_trace(
            predicted_items=predicted_items,
            actual_items=actual_rows,
            tick_index=int(tick_index),
        )
        touched_labels = []
        update_labels = sorted(set(trace.get("matched_labels", []) or []) | set(trace.get("missed_predicted_labels", []) or []) | set(trace.get("unexpected_labels", []) or []))
        for label in update_labels[: self.prediction_validation_update_limit]:
            entry = self._entries.get(str(label))
            if entry is None:
                continue
            self._touch_entry(entry)
            update = self._energy_updater.update_entry_from_trace(entry, label=str(label), trace=trace)
            entry.provenance.append(f"prediction_validation@{tick_index}:{update['role']}")
            entry.provenance = entry.provenance[-12:]
            entry.last_updated_tick = int(tick_index)
            touched_labels.append(update)
        residual_update = self._residual_tracker.ingest_prediction_trace(trace)
        trace["energy_updates"] = touched_labels[: max(1, self.r_state_items_per_head)]
        trace["residual_update"] = residual_update
        self._last_prediction_trace = trace
        self._pending_prediction_items = []
        return trace

    def apply_memory_bootstrap(self, snapshot: dict, *, tick_index: int) -> None:
        for item in snapshot.get("items", []) or []:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            entry = self._entries.get(label)
            if entry is None:
                entry = PoolEntry(
                    sa_label=label,
                    display_text=str(item.get("display_text", label) or label),
                    family=str(item.get("family", "bootstrap") or "bootstrap"),
                    source_type="memory_bootstrap",
                )
                entry.last_updated_tick = int(tick_index)
                self._entries[label] = entry
                self._entry_order.append(label)
            self._touch_entry(entry)
            # PHASE1_MINIMAL: coarse bootstrap heuristic until memory-only
            # recovery semantics are made explicit.
            entry.virtual_energy = max(entry.virtual_energy, self.bootstrap_virtual_energy * float(item.get("real_energy", 1.0) or 0.0))
            entry.anchor_meta["bootstrap_memory_id"] = snapshot.get("memory_id", "")
            entry.provenance.append(f"bootstrap@{tick_index}")
            entry.provenance = entry.provenance[-12:]
            entry.last_updated_tick = int(tick_index)
            entry.refresh_pressure()

    def select_focus(self, focus_labels: list[str]) -> None:
        unique_labels: list[str] = []
        for label in focus_labels:
            clean = str(label or "")
            if clean and clean not in unique_labels:
                unique_labels.append(clean)
        for label in unique_labels:
            entry = self._entries.get(label)
            if entry is None:
                continue
            self._touch_entry(entry)
            entry.attention_gain = float(entry.attention_gain) + self.focus_boost
            entry.fatigue = float(entry.fatigue) + self.focus_fatigue_step
            entry.anchor_meta["is_focus"] = True
            entry.last_updated_tick = int(self._tick_index)
            entry.refresh_pressure()
        next_focus = set(unique_labels)
        for label in list(self._current_focus_labels - next_focus):
            entry = self._entries.get(label)
            if entry is not None:
                entry.anchor_meta["is_focus"] = False
        self._current_focus_labels = next_focus

    def read_r_state(self, *, items_per_head: int | None = None, head_limit: int | None = None) -> dict:
        """
        Fixed-budget multi-head readout.

        This is the correct query source for fast-system recall in the HDB-V2 sense.
        `query_view()` remains an observability view only.
        """

        items_per_head = max(1, int(self.r_state_items_per_head if items_per_head is None else items_per_head))
        head_limit_value = max(1, int(self.r_state_head_limit if head_limit is None else head_limit))

        row_cache: dict[str, dict] = {}

        def _row_for_label(label: str) -> dict | None:
            if label in row_cache:
                return row_cache[label]
            entry = self._entries.get(label)
            if entry is None:
                return None
            self._touch_entry(entry)
            row = entry.as_dict(include_meta=False)
            current_source_types = sorted(self._current_source_types_by_label.get(label, set()))
            row["current_source_types"] = current_source_types
            row["current_tick_item"] = bool(current_source_types)
            row["query_weight"] = (
                max(
                    0.0,
                    row["real_energy"] * 0.95
                    + max(0.0, row["cognitive_pressure"]) * 0.85
                    + row["virtual_energy"] * 0.22
                    + row["attention_gain"] * 0.42
                    - row["fatigue"] * 0.45,
                )
            )
            row["attention_score"] = (
                max(
                    0.0,
                    row["cognitive_pressure"] * 1.0
                    + row["real_energy"] * 0.35
                    + row["attention_gain"] * 0.55
                    + row["virtual_energy"] * 0.12
                    - row["fatigue"] * 0.60,
                )
            )
            row_cache[label] = row
            return row

        def _dedup(rows: list[dict]) -> list[dict]:
            seen = set()
            kept = []
            for row in rows:
                label = str(row.get("sa_label", "") or "")
                if not label or label in seen:
                    continue
                seen.add(label)
                kept.append(row)
            return kept

        def _unique_labels(labels: list[str]) -> list[str]:
            seen = set()
            kept = []
            for label in labels:
                clean = str(label or "")
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                kept.append(clean)
            return kept

        heads: list[dict] = []

        # head_recent: current external evidence first, then recent external
        # context. Action feedback is an internal consequence signal; it may
        # live in anchors/residuals/successors, but must not masquerade as the
        # current sensory/text input.
        current_external_labels = _unique_labels(list(reversed(list(self._current_external_labels))))
        current_external_set = set(current_external_labels)
        recent_context_labels = [
            lab
            for lab in list(reversed(list(self._recent_external)[-max(1, self.recent_external_limit) :]))
            if str(lab or "") not in current_external_set
        ]
        recent_labels = _unique_labels(current_external_labels + recent_context_labels)
        # IMPORTANT:
        # Do not sort purely by energy here, otherwise long-lived anchors can starve
        # current-tick external evidence (breaking long-distance successor learning).
        # Keep newest-first, then dedup + cap.
        recent_rows = [r for r in (_row_for_label(lab) for lab in recent_labels) if isinstance(r, dict)]
        # Different slices of the same bounded queue let a large current
        # stimulus contribute 1024-level unique query evidence without scanning
        # the whole pool.
        for chunk_index in range(0, max(1, min(4, int(head_limit_value)))):
            start = chunk_index * items_per_head
            stop = start + items_per_head
            chunk = recent_rows[start:stop]
            if not chunk and chunk_index > 0:
                continue
            head_id = "head_recent" if chunk_index == 0 else f"head_recent_context_{chunk_index}"
            heads.append({"head_id": head_id, "items": chunk})

        # head_anchor: hot anchors.
        anchor_labels = _unique_labels(list(reversed(list(self._hot_anchor)[-max(1, self.hot_anchor_limit) :])))
        anchor_rows = [r for r in (_row_for_label(lab) for lab in anchor_labels) if isinstance(r, dict)]
        anchor_rows.sort(key=lambda item: (-float(item.get("attention_score", 0.0) or 0.0), -float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        heads.append({"head_id": "head_anchor", "items": _dedup(anchor_rows)[:items_per_head]})
        if len(anchor_rows) > items_per_head:
            heads.append({"head_id": "head_anchor_context", "items": _dedup(anchor_rows[items_per_head : items_per_head * 3])[:items_per_head]})

        # head_prediction: predicted virtual-energy entries recently injected.
        pred_labels = _unique_labels(list(reversed(list(self._prediction_slot)[-max(1, items_per_head * 4) :])))
        pred_rows = [r for r in (_row_for_label(lab) for lab in pred_labels) if isinstance(r, dict)]
        pred_rows.sort(key=lambda item: (-float(item.get("virtual_energy", 0.0) or 0.0), -float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        heads.append({"head_id": "head_prediction", "items": _dedup(pred_rows)[:items_per_head]})

        # head_residual: unresolved prediction/suppression bucket.
        residual_rows: list[dict] = []
        for payload in self._residual_tracker.items(limit=items_per_head * 2):
            lab = str(payload.get("sa_label", "") or "")
            row = _row_for_label(lab)
            if not row:
                continue
            row = dict(row)
            # residual gets a bounded boost so unresolved mismatch is not starved.
            row["query_weight"] = float(row.get("query_weight", 0.0) or 0.0) + float(payload.get("residual_boost", 0.1) or 0.1)
            row["residual_reason"] = str(payload.get("last_reason", "") or "")
            row["unresolved_mass"] = float(payload.get("unresolved_mass", 0.0) or 0.0)
            residual_rows.append(row)
        residual_rows.sort(key=lambda item: (-float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        heads.append({"head_id": "head_residual", "items": _dedup(residual_rows)[:items_per_head]})

        # head_global: best-effort top by query_weight without scanning:
        # We approximate via union of head_recent/head_anchor/head_prediction plus a bounded sample
        # from maintenance cursor (already touched).
        global_candidates: list[dict] = []
        for head in heads:
            global_candidates.extend(list(head.get("items", []) or []))

        # Add a bounded sample from the entry keys ring (maintenance budget sized).
        extra_labels = []
        if self._entry_order:
            keys = self._entry_order
            start = int(self._maintenance_cursor) % len(keys)
            take = min(max(8, items_per_head), max(0, self.maintenance_budget))
            for i in range(0, take):
                extra_labels.append(keys[(start + i) % len(keys)])
        for lab in extra_labels:
            row = _row_for_label(lab)
            if row:
                global_candidates.append(row)
        global_candidates = _dedup(global_candidates)
        global_candidates.sort(key=lambda item: (-float(item.get("query_weight", 0.0) or 0.0), -float(item.get("cognitive_pressure", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        heads.append({"head_id": "head_global", "items": global_candidates[:items_per_head]})

        # Enforce head limit.
        candidate_heads = heads
        heads = candidate_heads[: max(1, int(head_limit_value))]
        merged_preview = []
        seen = set()
        for head in heads:
            for row in head.get("items", []) or []:
                lab = str(row.get("sa_label", "") or "")
                if lab and lab not in seen:
                    seen.add(lab)
                    merged_preview.append(lab)
        return {
            "schema_id": "r_state_snapshot/v1",
            "schema_version": "1.0",
            "tick_index": int(self._tick_index),
            "head_count": len(heads),
            "items_per_head": int(items_per_head),
            "head_limit": int(head_limit_value),
            "heads": heads,
            "merged_preview": merged_preview[: max(1, items_per_head * len(heads))],
            "total_pool_size": len(self._entries),
            "available_head_ids": [head.get("head_id", "") for head in candidate_heads],
            "maintenance_cursor": int(self._maintenance_cursor),
            "prediction_trace": self.prediction_trace(),
            "residual_summary": self.residual_summary(limit=min(8, items_per_head)),
        }

    def query_view(self) -> list[dict]:
        # White-box view only: bounded. It may traverse the pool, but query_limit is small.
        # IMPORTANT: This is not the recall mainline query source.
        rows = []
        for entry in self._entries.values():
            self._touch_entry(entry)
            row = entry.as_dict()
            # PHASE1_MINIMAL: provisional query weighting formula.
            row["query_weight"] = (
                max(
                    0.0,
                    row["real_energy"] * 0.95
                    + max(0.0, row["cognitive_pressure"]) * 0.85
                    + row["virtual_energy"] * 0.22
                    + row["attention_gain"] * 0.42
                    - row["fatigue"] * 0.45,
                )
            )
            rows.append(row)
        rows.sort(key=lambda item: (-float(item["query_weight"]), -float(item["cognitive_pressure"]), str(item["sa_label"])))
        return rows[: self.query_limit]

    def attention_view(self) -> list[dict]:
        # White-box view only: bounded. It may traverse the pool, but the runtime should
        # prefer using `read_r_state()` heads for bounded attention candidates later.
        rows = []
        for entry in self._entries.values():
            self._touch_entry(entry)
            row = entry.as_dict()
            # PHASE1_MINIMAL: provisional attention weighting formula.
            row["attention_score"] = (
                max(
                    0.0,
                    row["cognitive_pressure"] * 1.0
                    + row["real_energy"] * 0.35
                    + row["attention_gain"] * 0.55
                    + row["virtual_energy"] * 0.12
                    - row["fatigue"] * 0.60,
                )
            )
            rows.append(row)
        rows.sort(key=lambda item: (-float(item["attention_score"]), str(item["sa_label"])))
        return rows[: self.query_limit]

    def prediction_trace(self) -> dict:
        trace = dict(self._last_prediction_trace)
        trace["cstar_budget_trace"] = self.cstar_budget_trace()
        return trace

    def cstar_budget_trace(self) -> dict:
        return dict(self._last_cstar_budget_trace)

    def residual_summary(self, *, limit: int | None = None) -> dict:
        return self._residual_tracker.snapshot(limit=limit)

    def energy_flow_trace(
        self,
        *,
        items: list[dict] | None = None,
        r_state: dict | None = None,
        memory_write_items: list[dict] | None = None,
        limit: int = 8,
    ) -> dict:
        """
        Read-only explanation of the current dual-energy field.

        AP's state pool must stay free to evolve; this method does not teach,
        decay, boost, or select anything. It only turns the current bounded
        views into a theory-facing map: real energy, virtual energy, pressure,
        residual, and the separate contracts of snapshot/R_state/memory-write.
        """

        rows = list(items or [])
        if not rows:
            # Avoid calling snapshot() here: snapshot() itself attaches an
            # energy_flow trace, so an empty view would recurse forever. When
            # the caller does not provide a view, build the explanation from a
            # bounded direct entry read instead.
            entries = sorted(self._entries.values(), key=self._entry_sort_key)
            rows = [entry.as_dict() for entry in entries[: max(1, int(limit))]]
        return build_energy_flow_trace(
            items=rows,
            tick_index=int(self._tick_index),
            prediction_trace=self.prediction_trace(),
            residual_summary=self.residual_summary(limit=limit),
            r_state=r_state,
            memory_write_items=memory_write_items,
            limit=limit,
        )

    def _entry_total_energy(self, entry: PoolEntry) -> float:
        return float(entry.real_energy or 0.0) + float(entry.virtual_energy or 0.0)

    def _entry_sort_key(self, entry: PoolEntry) -> tuple:
        return (
            -self._entry_total_energy(entry),
            -float(entry.cognitive_pressure or 0.0),
            str(entry.sa_label or ""),
        )

    def _entry_bbox_norm(self, entry: PoolEntry) -> list[float]:
        meta = dict(entry.anchor_meta or {})
        bbox = list(meta.get("bbox_norm", []) or [])
        if len(bbox) >= 4:
            return [_round4(max(0.0, min(1.0, float(value or 0.0)))) for value in bbox[:4]]
        numeric = dict(entry.numeric_features or {})
        spatial = list(numeric.get("vision.spatial", []) or [])
        if len(spatial) >= 4:
            return [_round4(max(0.0, min(1.0, float(value or 0.0)))) for value in spatial[:4]]
        return []

    def _is_glyph_like_visual_entry(self, entry: PoolEntry) -> bool:
        if str(entry.family or "") != "vision_object":
            return False
        label = str(entry.sa_label or "")
        meta = dict(entry.anchor_meta or {})
        return (
            "glyph_slice" in label
            or "glyph_slice" in str(meta.get("object_anchor_id", "") or "")
            or str(meta.get("proposal_kind", "") or "") == "glyph_slice"
        )

    def _current_glyph_spatial_diversity_entries(self, entries: list[PoolEntry], *, limit: int = 4) -> list[PoolEntry]:
        """
        View-only coverage for foveated reading.

        The action planner cannot learn a left-to-right gaze/readback habit if
        the bounded snapshot drops the quiet start of a visible glyph row. This
        helper preserves spatially diverse *current tick* glyph-like visual
        objects for the planner view. It deliberately ignores character labels,
        teacher references, expected text, and answers.
        """

        candidates: list[tuple[float, float, PoolEntry]] = []
        for entry in entries:
            if int(entry.last_seen_tick) != int(self._tick_index):
                continue
            if not self._is_glyph_like_visual_entry(entry):
                continue
            bbox = self._entry_bbox_norm(entry)
            if len(bbox) < 4:
                continue
            candidates.append((float(bbox[0]), float(bbox[1]), entry))
        if not candidates:
            return []
        candidates.sort(key=lambda row: (row[1], row[0], str(row[2].sa_label or "")))
        selected: list[PoolEntry] = []
        selected_labels: set[str] = set()

        def add(entry: PoolEntry) -> None:
            if len(selected) >= max(1, int(limit)):
                return
            label = str(entry.sa_label or "")
            if not label or label in selected_labels:
                return
            selected.append(entry)
            selected_labels.add(label)

        # Preserve the visual row's start, end, and a few interior representatives.
        # This is coverage, not a scan command; competition still decides what AP
        # attends to and whether it emits any low-grain character action.
        ordered = sorted(candidates, key=lambda row: (row[0], row[1], str(row[2].sa_label or "")))
        add(ordered[0][2])
        if len(ordered) > 1:
            add(ordered[-1][2])
        if len(ordered) > 2:
            add(ordered[len(ordered) // 2][2])
        if len(ordered) > 3:
            add(ordered[max(1, len(ordered) // 3)][2])
        return selected

    def _entry_memory_row(self, entry: PoolEntry) -> dict:
        family = str(entry.family or "")
        source_type = str(entry.source_type or "")
        include_meta = bool(entry.anchor_meta) and (
            family
            in {
                "action",
                "action_control",
                "action_feedback",
                "text_action",
                "cognitive_feeling",
                "time_feeling",
                "rhythm_feeling",
                "expectation_pressure",
                "vision",
                "vision_dynamic",
                "vision_channel",
                "vision_object",
                "audio",
                "audio_channel",
                "audio_event",
            }
            or self._has_process_anchor_meta(entry.anchor_meta)
            or source_type
            in {
                "action_selection",
                "action_control",
                "action_feedback",
                "text_action",
                "cognitive_feeling",
                "time_feeling",
                "rhythm_feeling",
                "expectation_pressure",
                "external_teacher",
                "vision_bridge",
                "vision_bridge_dynamic",
                "audio_bridge",
                "vision_numeric",
                "audio_numeric",
            }
            or source_type.endswith("_external_teacher")
        )
        if entry.numeric_features:
            include_meta = True
        return entry.as_dict(include_meta=include_meta)

    def _is_external_evidence_entry(self, entry: PoolEntry) -> bool:
        src = str(entry.source_type or "")
        if src == "external_text":
            return True
        if src == "external_teacher" or src.endswith("_external_teacher"):
            return True
        if src.startswith("vision_bridge"):
            return True
        if src.startswith("audio_bridge"):
            return True
        if src == "vision_numeric":
            return True
        if src == "audio_numeric":
            return True
        return False

    def rows_for_labels(self, labels: list[str]) -> list[dict]:
        """
        Bounded direct lookup for slow-system focus continuation.

        Slow recall should develop explicit focus objects instead of requiring a
        whole-pool white-box snapshot. This method touches only requested labels.
        """

        rows: list[dict] = []
        seen = set()
        for label in labels or []:
            clean = str(label or "")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            entry = self._entries.get(clean)
            if entry is None:
                continue
            self._touch_entry(entry)
            row = entry.as_dict(include_meta=False)
            current_source_types = sorted(self._current_source_types_by_label.get(clean, set()))
            row["current_source_types"] = current_source_types
            row["current_tick_item"] = bool(current_source_types)
            row["query_weight"] = (
                max(
                    0.0,
                    row["real_energy"] * 0.95
                    + max(0.0, row["cognitive_pressure"]) * 0.85
                    + row["virtual_energy"] * 0.22
                    + row["attention_gain"] * 0.42
                    - row["fatigue"] * 0.45,
                )
            )
            row["attention_score"] = (
                max(
                    0.0,
                    row["cognitive_pressure"] * 1.0
                    + row["real_energy"] * 0.35
                    + row["attention_gain"] * 0.55
                    + row["virtual_energy"] * 0.12
                    - row["fatigue"] * 0.60,
                )
            )
            rows.append(row)
        return rows

    def snapshot(self) -> dict:
        """
        Snapshot is a white-box "cognitive field view" for memory write + observability.

        IMPORTANT:
        - The pool itself is the source of truth, but `snapshot()` is a bounded view.
        - We must avoid "control" virtual-energy injections completely hiding external evidence
          (vision/audio) when `snapshot_limit` is small.

        Design:
        1) Take a primary top-N by a stable relevance score (still includes virtual energy).
        2) Enforce minimum quotas for key external families if they exist in the pool.
           This keeps multimodal evidence auditable and prevents accidental evictions.

        This is intentionally a view-layer policy: it does not change underlying energies.
        """

        limit = max(1, int(self.snapshot_limit))
        entries = list(self._entries.values())
        ordered_entries = nsmallest(limit, entries, key=self._entry_sort_key)
        selected_entries = list(ordered_entries[:limit])
        selected_labels = {str(entry.sa_label or "") for entry in selected_entries}

        # External evidence quotas (view-only): keep at least K items for a family if the
        # pool currently contains that family at all.
        quotas = {
            "vision": 1,
            "vision_channel": 1,
            "vision_object": 1,
            "audio": 1,
            "audio_channel": 1,
            "audio_event": 1,
            # Time/rhythm feelings are low-mass but foundational subjective
            # channels. Parallel actions can create many feedback rows in one
            # tick; this view-only quota keeps active temporal feelings visible
            # for trace/tests without changing their underlying energy.
            "time_feeling": 1,
            "rhythm_feeling": 1,
        }

        def inject_entry(extra: PoolEntry, *, protected_families: set[str]) -> None:
            label = str(extra.sa_label or "")
            if not label or label in selected_labels:
                return
            if len(selected_entries) < limit:
                selected_entries.append(extra)
                selected_labels.add(label)
                return

            replace_idx = None
            # pick the worst non-protected item; if all are protected, fall back to worst overall
            for idx in range(len(selected_entries) - 1, -1, -1):
                fam = str(selected_entries[idx].family or "")
                if fam not in protected_families:
                    replace_idx = idx
                    break
            if replace_idx is None:
                replace_idx = len(selected_entries) - 1
            old_label = str(selected_entries[replace_idx].sa_label or "")
            selected_entries[replace_idx] = extra
            selected_labels.discard(old_label)
            selected_labels.add(label)

        protected_families = set(quotas.keys())
        for family, min_count in quotas.items():
            if min_count <= 0:
                continue
            family_entries = nsmallest(
                int(min_count),
                (entry for entry in entries if str(entry.family or "") == family),
                key=self._entry_sort_key,
            )
            if not family_entries:
                continue
            have = len([entry for entry in selected_entries if str(entry.family or "") == family])
            need = max(0, int(min_count) - have)
            if need <= 0:
                continue

            candidates = [entry for entry in family_entries if str(entry.sa_label or "") not in selected_labels]
            inject = candidates[:need]
            if not inject:
                continue

            # Replace the lowest-ranked items that are not from the protected families.
            for extra in inject:
                inject_entry(extra, protected_families=protected_families)

        # View-only object diversity for embodied attention:
        # A snapshot used by the action planner must expose several currently
        # seen visual objects, otherwise gaze can only compete over whichever
        # single object survived the small observability top-N. This does not
        # mutate energy and does not script scan order; it simply preserves the
        # live external candidates needed for humanlike eye movement.
        current_visual_objects = nsmallest(
            6,
            (
                entry
                for entry in entries
                if str(entry.family or "") == "vision_object"
                and int(entry.last_seen_tick) == int(self._tick_index)
            ),
            key=self._entry_sort_key,
        )
        for extra in current_visual_objects:
            inject_entry(extra, protected_families=protected_families)
        for extra in self._current_glyph_spatial_diversity_entries(entries, limit=4):
            inject_entry(extra, protected_families=protected_families)

        # Primary reconstruction anchors are field/focus SA, not raw assets. They are
        # low-energy by design, but observatory and memory-audit views need one stable
        # representative so the inner visual/audio replay does not get hidden by
        # higher-energy feelings or action traces.
        for label in ("vision::field::color_grid", "audio::focus::waveform_slice"):
            entry = self._entries.get(label)
            if entry is not None:
                inject_entry(entry, protected_families=protected_families)
        commit_ready_entry = self._entries.get("state::commit_ready")
        if commit_ready_entry is not None:
            inject_entry(commit_ready_entry, protected_families=protected_families)

        selected_entries.sort(key=self._entry_sort_key)
        snapshot_items = [entry.as_dict() for entry in selected_entries[:limit]]
        energy_flow = self.energy_flow_trace(items=snapshot_items, limit=8)
        return {
            "tick_index": self._tick_index,
            "items": snapshot_items,
            "prediction_trace": self.prediction_trace(),
            "residual_summary": self.residual_summary(limit=8),
            "energy_flow": energy_flow,
        }

    def snapshot_for_memory_write(self) -> dict:
        """
        A snapshot specifically for long-term memory write.

        Theory alignment / why this exists:
        - `snapshot()` is a bounded observability view and can be evicted by high-energy
          derived channels (time/feelings/actions).
        - Memory write MUST NOT drop current-tick external evidence (text/vision/audio),
          otherwise successor transitions (Cn) cannot learn/predict long-distance dependencies.

        Policy:
        - Start from the normal snapshot view.
        - Force-include all "external evidence" items that were observed at this tick.
          (Currently: `source_type` in {external_text, external_teacher,
           vision_bridge*, audio_bridge*, vision_numeric, audio_numeric} and
           `last_seen_tick == current_tick`.)
        - Keep within `snapshot_limit` by evicting the lowest-ranked non-external items first.

        This is still a view-layer policy: it never mutates underlying energies.
        """

        limit = max(1, int(self.memory_snapshot_limit))

        selected_entries: list[PoolEntry] = []
        selected_labels: set[str] = set()

        def add_label(label: str) -> None:
            clean = str(label or "")
            if not clean or clean in selected_labels or len(selected_entries) >= limit:
                return
            entry = self._entries.get(clean)
            if entry is None:
                return
            self._touch_entry(entry)
            selected_labels.add(clean)
            selected_entries.append(entry)

        # 1) Current-tick external evidence is mandatory and already bounded by
        # the sensor budget. This is the most important signal for state->state
        # successor learning and avoids a full-pool scan.
        for label in self._current_external_labels:
            add_label(label)
            if len(selected_entries) >= limit:
                return {"tick_index": self._tick_index, "items": [self._entry_memory_row(entry) for entry in selected_entries[:limit]]}

        # 2) Fill with bounded hot/recent/prediction/focus candidates. These are
        # incremental caches, so runtime cost is weakly related to total pool size.
        candidate_labels: list[str] = []
        candidate_labels.extend(reversed(self._recent_external[-self.recent_external_limit :]))
        candidate_labels.extend(reversed(self._hot_anchor[-self.hot_anchor_limit :]))
        candidate_labels.extend(reversed(self._prediction_slot[-max(8, self.r_state_items_per_head * 2) :]))
        candidate_labels.extend(self._current_focus_labels)
        if "state::commit_ready" not in selected_labels:
            candidate_labels.append("state::commit_ready")

        candidates: list[PoolEntry] = []
        seen_candidates = set(selected_labels)
        for label in candidate_labels:
            clean = str(label or "")
            if not clean or clean in seen_candidates:
                continue
            entry = self._entries.get(clean)
            if entry is None:
                continue
            self._touch_entry(entry)
            seen_candidates.add(clean)
            candidates.append(entry)

        candidates.sort(key=self._entry_sort_key)
        for entry in candidates:
            if len(selected_entries) >= limit:
                break
            label = str(entry.sa_label or "")
            if not label or label in selected_labels:
                continue
            selected_labels.add(label)
            selected_entries.append(entry)

        return {"tick_index": self._tick_index, "items": [self._entry_memory_row(entry) for entry in selected_entries[:limit]]}
