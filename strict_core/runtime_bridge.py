from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.defaults import MemoryConfig, StatePoolConfig
from core.state_pool.state_pool import DualEnergyStatePool
from memory.store.memory_store import MemoryStore
from strict_core.common import action_feedback_item, action_item


def make_state_pool(config: StatePoolConfig | None = None) -> DualEnergyStatePool:
    cfg = config or StatePoolConfig(snapshot_limit=32, memory_snapshot_limit=256, r_state_items_per_head=64)
    return DualEnergyStatePool(
        real_decay=cfg.real_decay,
        virtual_decay=cfg.virtual_decay,
        attention_gain_decay=cfg.attention_gain_decay,
        fatigue_decay=cfg.fatigue_decay,
        prune_threshold=cfg.prune_threshold,
        query_limit=cfg.query_limit,
        snapshot_limit=cfg.snapshot_limit,
        memory_snapshot_limit=cfg.memory_snapshot_limit,
        r_state_head_limit=cfg.r_state_head_limit,
        r_state_items_per_head=cfg.r_state_items_per_head,
        maintenance_budget=cfg.maintenance_budget,
        recent_external_limit=cfg.recent_external_limit,
        hot_anchor_limit=cfg.hot_anchor_limit,
        prediction_validation_actual_limit=cfg.prediction_validation_actual_limit,
        prediction_validation_update_limit=cfg.prediction_validation_update_limit,
        focus_boost=cfg.focus_boost,
        focus_fatigue_step=cfg.focus_fatigue_step,
        prediction_fatigue_enabled=cfg.prediction_fatigue_enabled,
        prediction_fatigue_min_mass=cfg.prediction_fatigue_min_mass,
        prediction_fatigue_ratio=cfg.prediction_fatigue_ratio,
        prediction_fatigue_gain=cfg.prediction_fatigue_gain,
        prediction_fatigue_max_step=cfg.prediction_fatigue_max_step,
        cstar_trace_top_labels=cfg.cstar_trace_top_labels,
        bootstrap_virtual_energy=cfg.bootstrap_virtual_energy,
    )


def make_memory(config: MemoryConfig | None = None) -> MemoryStore:
    cfg = config or MemoryConfig(
        recall_top_k=8,
        max_snapshots_per_kind=512,
        candidate_limit=128,
        core_item_limit=256,
        query_feature_limit=256,
        numeric_candidate_limit=64,
        numeric_top_k_per_channel=24,
        index_jobs_per_tick=4,
    )
    return MemoryStore(
        recall_top_k=cfg.recall_top_k,
        predict_top_k=cfg.predict_top_k,
        prediction_energy_scale=cfg.prediction_energy_scale,
        max_snapshots_per_kind=cfg.max_snapshots_per_kind,
        candidate_limit=cfg.candidate_limit,
        core_item_limit=cfg.core_item_limit,
        query_feature_limit=cfg.query_feature_limit,
        posting_label_token_limit=cfg.posting_label_token_limit,
        posting_display_token_limit=cfg.posting_display_token_limit,
        posting_bigram_token_limit=cfg.posting_bigram_token_limit,
        posting_sequence_token_limit=cfg.posting_sequence_token_limit,
        vector_token_limit=cfg.vector_token_limit,
        scoring_candidate_limit=cfg.scoring_candidate_limit,
        learned_rerank_limit=cfg.learned_rerank_limit,
        state_query_signature_token_limit=cfg.state_query_signature_token_limit,
        numeric_enabled=cfg.numeric_enabled,
        numeric_dim=cfg.numeric_dim,
        numeric_candidate_limit=cfg.numeric_candidate_limit,
        numeric_top_k_per_channel=cfg.numeric_top_k_per_channel,
        numeric_weight=cfg.numeric_weight,
        relation_enabled=cfg.relation_enabled,
        relation_token_limit=cfg.relation_token_limit,
        relation_event_limit=cfg.relation_event_limit,
        relation_context_limit=cfg.relation_context_limit,
        relation_score_weight=cfg.relation_score_weight,
        relation_focus_score_weight=cfg.relation_focus_score_weight,
        temporal_applicability_enabled=cfg.temporal_applicability_enabled,
        temporal_tick_seconds=cfg.temporal_tick_seconds,
        temporal_fatigue_window_ticks=cfg.temporal_fatigue_window_ticks,
        temporal_fatigue_strength=cfg.temporal_fatigue_strength,
        temporal_recent_gain_window_ticks=cfg.temporal_recent_gain_window_ticks,
        temporal_recent_gain=cfg.temporal_recent_gain,
        temporal_long_half_life_ticks=cfg.temporal_long_half_life_ticks,
        temporal_floor=cfg.temporal_floor,
        index_jobs_per_tick=cfg.index_jobs_per_tick,
        online_enabled=True,
        online_dim=32,
        online_token_limit=2048,
        online_min_support_to_promote=2,
        online_per_tick_update_limit=8,
        online_scoring_token_limit=256,
        learned_weight=0.28,
        transition_learned_weight=0.18,
        ann_enabled=False,
    )


@dataclass
class StrictTickTrace:
    tick_index: int
    visible_observation: list[dict]
    state_pool_snapshot: dict
    memory_rows: list[dict]


class StrictRuntimeBridge:
    """Small bridge that reuses APV2.1 core state/memory interfaces.

    This is intentionally narrower than the full APV21Runtime tick.  The proof
    target here is whether strict visible items can pass through the same
    DualEnergyStatePool and MemoryStore surfaces that APV2.1 uses elsewhere.
    """

    def __init__(self) -> None:
        self.state_pool = make_state_pool()
        self.memory = make_memory()
        self.tick_index = -1

    def begin_case(self, state_items: list[dict], *, memory_kind: str = "strict_state") -> StrictTickTrace:
        self.tick_index += 1
        self.state_pool.begin_tick(self.tick_index)
        self.state_pool.apply_external_items(list(state_items), tick_index=self.tick_index)
        snapshot = self.state_pool.snapshot_for_memory_write()
        query_items = list(snapshot.get("items", []) or [])
        memory_rows = self.memory.recall(query_items, memory_kind=memory_kind, top_k=64)
        return StrictTickTrace(
            tick_index=self.tick_index,
            visible_observation=list(state_items),
            state_pool_snapshot=snapshot,
            memory_rows=memory_rows,
        )

    def commit_action(
        self,
        *,
        state_items: list[dict],
        action_id: str,
        reason: str,
        case_id: str,
        feedback: dict | None,
        learning_enabled: bool,
        memory_kind: str = "strict_state",
    ) -> dict:
        if not learning_enabled or feedback is None:
            return {"memory_written": False, "memory_id": None, "items_written": 0}
        action = action_item(action_id=action_id, tick_index=self.tick_index, reason=reason)
        feedback_item = action_feedback_item(
            action_id=action_id,
            feedback=feedback,
            tick_index=self.tick_index,
            case_id=case_id,
            source="strict_training_feedback",
        )
        items = list(state_items) + [action, feedback_item]
        self.memory.write_snapshot(
            tick_index=self.tick_index,
            memory_kind=memory_kind,
            items=items,
            focus_labels=[str(action_id), str(feedback_item["sa_label"])],
            source_text="strict action-feedback training snapshot",
        )
        self.memory.process_pending_index_jobs(budget=16, include_heavy=True)
        return {
            "memory_written": True,
            "memory_id": f"mem-{self.memory._next_id - 1}",
            "items_written": len(items),
        }

    def load_package(self, package: dict, *, memory_kind: str = "strict_state") -> dict:
        loaded = 0
        for row in list(package.get("experience_rows", []) or []):
            items = list(row.get("items", []) or [])
            focus = list(row.get("focus_labels", []) or [])
            if not items:
                continue
            self.tick_index += 1
            self.memory.write_snapshot(
                tick_index=self.tick_index,
                memory_kind=memory_kind,
                items=items,
                focus_labels=focus,
                source_text="strict experience package reload",
            )
            loaded += 1
        self.memory.process_pending_index_jobs(budget=64, include_heavy=True)
        return {"loaded_rows": loaded, "memory_kind": memory_kind}

    def export_package(self, *, memory_kind: str = "strict_state", package_id: str) -> dict:
        rows: list[dict[str, Any]] = []
        for snapshot in self.memory._recent_by_kind.get(memory_kind, []):
            rows.append(
                {
                    "tick_index": int(snapshot.get("tick_index", -1) or -1),
                    "items": list(snapshot.get("items", []) or []),
                    "focus_labels": list(snapshot.get("focus_labels", []) or []),
                }
            )
        return {
            "schema_id": "strict_experience_package/v1",
            "package_id": str(package_id),
            "memory_kind": str(memory_kind),
            "experience_rows": rows,
            "not_claiming": [
                "complete_math_ability",
                "open_world_semantics",
                "independent_vision_or_asr",
            ],
        }
