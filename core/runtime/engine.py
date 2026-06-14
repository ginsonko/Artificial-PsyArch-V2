from __future__ import annotations

import gc
from heapq import nsmallest
from pathlib import Path
from time import perf_counter

from channels.cognitive_feelings.channel import CognitiveFeelingChannel
from channels.expectation_pressure import BAnchorExpectationVerifier, ExpectationPressureChannel
from channels.rhythm.channel import RhythmChannel
from channels.runtime_load import RuntimeLoadFeelingChannel
from channels.task_feeling import TaskFeelingChannel
from channels.time.channel import TimeFeelingChannel
from config.defaults import RuntimeConfig
from core.action import ActionConsequenceEvaluator, ActionConsequencePlanner, ActionControlEffectRouter, AuditoryBandActuator, SafetyGate, TextActionActuator, VisualGazeActuator
from core.attention.selector import AttentionSelector
from core.cognition.sa_registry import SARegistry
from core.emotion import EmotionModulator
from core.innate import InnateCodingEngine
from core.learning import InnateLearningEventRouter
from learning_events import LearningEventBuilder
from core.runtime.budget_controller import RuntimeBudgetController
from core.state_pool.state_pool import DualEnergyStatePool
from core.tuner import AdaptiveTuner
from education.intervention import EducationInterventionBuffer
from memory.short_term.echo_buffer import ShortTermEchoBuffer
from memory.short_term.focus_buffer import FocusBuffer
from memory.short_term.focus_successor_bias import FocusSuccessorBias
from memory.short_term.memory_window import ShortTermMemoryWindow
from memory.short_term.slot_packet import ShortTermSlotPacketBuilder
from memory.assets import MultimodalAssetStore
from memory.store.memory_store import MemoryStore
from sensors.audio import LegacyAudioBridge, NativeAudioNumericSensor
from sensors.text.sensor import TextSensor
from sensors.vision import LegacyVisionBridge, NativeVisionNumericSensor

"""
PHASE1_MINIMAL_UPGRADED:
The runtime is still compact, but it is now split into explicit stage methods
and the slow-system query is built from a focus continuation buffer instead of
only the newest selected labels.
"""


def _round4(value: float) -> float:
    return round(float(value), 4)


class APV21Runtime:
    def __init__(self, *, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.text_sensor = TextSensor(budget_limit=self.config.text_sensor.budget_limit)
        self.vision_sensor = NativeVisionNumericSensor(
            max_objects=self.config.vision_sensor.max_objects,
            max_side=self.config.vision_sensor.max_side,
            preview_side=self.config.vision_sensor.preview_side,
        )
        self.audio_sensor = NativeAudioNumericSensor(
            max_samples=self.config.audio_sensor.max_samples,
            band_count=self.config.audio_sensor.band_count,
        )
        self.asset_store = MultimodalAssetStore(
            max_assets=self.config.multimodal_assets.max_assets,
            root_dir=str(Path(self.config.multimodal_assets.asset_root_dir)),
            persist_payloads=self.config.multimodal_assets.persist_payloads,
            keep_hot_payloads=self.config.multimodal_assets.keep_hot_payloads,
        )
        self.vision_bridge = LegacyVisionBridge()
        self.audio_bridge = LegacyAudioBridge()
        self.sa_registry = SARegistry(
            dynamic_phrase_min_observations=self.config.text_sensor.dynamic_phrase_min_observations,
            dynamic_phrase_max_len=self.config.text_sensor.dynamic_phrase_max_len,
            dynamic_phrase_scan_budget=self.config.text_sensor.dynamic_phrase_scan_budget,
            dynamic_phrase_emit_budget=self.config.text_sensor.dynamic_phrase_emit_budget,
        )
        self.state_pool = DualEnergyStatePool(
            real_decay=self.config.state_pool.real_decay,
            virtual_decay=self.config.state_pool.virtual_decay,
            attention_gain_decay=self.config.state_pool.attention_gain_decay,
            fatigue_decay=self.config.state_pool.fatigue_decay,
            prune_threshold=self.config.state_pool.prune_threshold,
            query_limit=self.config.state_pool.query_limit,
            snapshot_limit=self.config.state_pool.snapshot_limit,
            memory_snapshot_limit=self.config.state_pool.memory_snapshot_limit,
            r_state_head_limit=self.config.state_pool.r_state_head_limit,
            r_state_items_per_head=self.config.state_pool.r_state_items_per_head,
            maintenance_budget=self.config.state_pool.maintenance_budget,
            recent_external_limit=self.config.state_pool.recent_external_limit,
            hot_anchor_limit=self.config.state_pool.hot_anchor_limit,
            prediction_validation_actual_limit=self.config.state_pool.prediction_validation_actual_limit,
            prediction_validation_update_limit=self.config.state_pool.prediction_validation_update_limit,
            focus_boost=self.config.state_pool.focus_boost,
            focus_fatigue_step=self.config.state_pool.focus_fatigue_step,
            prediction_fatigue_enabled=self.config.state_pool.prediction_fatigue_enabled,
            prediction_fatigue_min_mass=self.config.state_pool.prediction_fatigue_min_mass,
            prediction_fatigue_ratio=self.config.state_pool.prediction_fatigue_ratio,
            prediction_fatigue_gain=self.config.state_pool.prediction_fatigue_gain,
            prediction_fatigue_max_step=self.config.state_pool.prediction_fatigue_max_step,
            cstar_trace_top_labels=self.config.state_pool.cstar_trace_top_labels,
            bootstrap_virtual_energy=self.config.state_pool.bootstrap_virtual_energy,
        )
        self.attention = AttentionSelector(
            focus_limit=self.config.attention.focus_limit,
            pressure_gain=self.config.attention.pressure_gain,
            attention_gain_weight=self.config.attention.attention_gain_weight,
            fatigue_weight=self.config.attention.fatigue_weight,
            continuation_bias=self.config.attention.continuation_bias,
            real_energy_weight=self.config.attention.real_energy_weight,
            virtual_energy_weight=self.config.attention.virtual_energy_weight,
        )
        self.memory = MemoryStore(
            recall_top_k=self.config.memory.recall_top_k,
            predict_top_k=self.config.memory.predict_top_k,
            prediction_energy_scale=self.config.memory.prediction_energy_scale,
            max_snapshots_per_kind=self.config.memory.max_snapshots_per_kind,
            candidate_limit=self.config.memory.candidate_limit,
            core_item_limit=self.config.memory.core_item_limit,
            query_feature_limit=self.config.memory.query_feature_limit,
            posting_label_token_limit=self.config.memory.posting_label_token_limit,
            posting_display_token_limit=self.config.memory.posting_display_token_limit,
            posting_bigram_token_limit=self.config.memory.posting_bigram_token_limit,
            posting_sequence_token_limit=self.config.memory.posting_sequence_token_limit,
            vector_token_limit=self.config.memory.vector_token_limit,
            scoring_candidate_limit=self.config.memory.scoring_candidate_limit,
            learned_rerank_limit=self.config.memory.learned_rerank_limit,
            state_query_signature_token_limit=self.config.memory.state_query_signature_token_limit,
            numeric_enabled=self.config.memory.numeric_enabled,
            numeric_dim=self.config.memory.numeric_dim,
            numeric_candidate_limit=self.config.memory.numeric_candidate_limit,
            numeric_top_k_per_channel=self.config.memory.numeric_top_k_per_channel,
            numeric_weight=self.config.memory.numeric_weight,
            relation_enabled=self.config.memory.relation_enabled,
            relation_token_limit=self.config.memory.relation_token_limit,
            relation_event_limit=self.config.memory.relation_event_limit,
            relation_context_limit=self.config.memory.relation_context_limit,
            relation_score_weight=self.config.memory.relation_score_weight,
            relation_focus_score_weight=self.config.memory.relation_focus_score_weight,
            temporal_applicability_enabled=self.config.memory.temporal_applicability_enabled,
            temporal_tick_seconds=self.config.memory.temporal_tick_seconds,
            temporal_fatigue_window_ticks=self.config.memory.temporal_fatigue_window_ticks,
            temporal_fatigue_strength=self.config.memory.temporal_fatigue_strength,
            temporal_fatigue_recovery_exponent=self.config.memory.temporal_fatigue_recovery_exponent,
            temporal_recent_gain_window_ticks=self.config.memory.temporal_recent_gain_window_ticks,
            temporal_recent_gain=self.config.memory.temporal_recent_gain,
            temporal_long_half_life_ticks=self.config.memory.temporal_long_half_life_ticks,
            temporal_floor=self.config.memory.temporal_floor,
            index_jobs_per_tick=self.config.memory.index_jobs_per_tick,
            long_term_recall_kinds=self.config.memory.long_term_recall_kinds,
            ann_enabled=True,
            online_enabled=self.config.online_embedding.enabled,
            online_dim=self.config.online_embedding.dim,
            online_token_limit=self.config.online_embedding.token_limit,
            online_min_support_to_promote=self.config.online_embedding.min_support_to_promote,
            online_per_tick_update_limit=self.config.online_embedding.per_tick_update_limit,
            online_scoring_token_limit=self.config.online_embedding.scoring_token_limit,
            learned_weight=self.config.online_embedding.learned_weight,
            learned_vector_candidate_weight=self.config.online_embedding.learned_vector_candidate_weight,
            transition_learned_weight=self.config.online_embedding.transition_learned_weight,
        )
        self.cognitive_feelings = CognitiveFeelingChannel(
            min_activation=self.config.cognitive_feelings.min_activation,
            surprise_gain=self.config.cognitive_feelings.surprise_gain,
            coherence_gain=self.config.cognitive_feelings.coherence_gain,
            dissonance_gain=self.config.cognitive_feelings.dissonance_gain,
            correctness_gain=self.config.cognitive_feelings.correctness_gain,
            grasp_gain=self.config.cognitive_feelings.grasp_gain,
            expectation_gain=self.config.cognitive_feelings.expectation_gain,
            pressure_gain=self.config.cognitive_feelings.pressure_gain,
        )
        self.task_feeling = TaskFeelingChannel(
            min_activation=self.config.task_feeling.min_activation,
            boredom_gain=self.config.task_feeling.boredom_gain,
            fulfillment_gain=self.config.task_feeling.fulfillment_gain,
        )
        self.runtime_load_feeling = RuntimeLoadFeelingChannel(
            enabled=self.config.runtime_load_feeling.enabled,
            min_activation=self.config.runtime_load_feeling.min_activation,
            complexity_gain=self.config.runtime_load_feeling.complexity_gain,
            simplicity_gain=self.config.runtime_load_feeling.simplicity_gain,
            target_load_ratio=self.config.runtime_load_feeling.target_load_ratio,
            ideal_load_ratio=self.config.runtime_load_feeling.ideal_load_ratio,
            state_item_soft_limit=self.config.runtime_load_feeling.state_item_soft_limit,
            r_state_item_soft_limit=self.config.runtime_load_feeling.r_state_item_soft_limit,
            attention_candidate_soft_limit=self.config.runtime_load_feeling.attention_candidate_soft_limit,
            pending_index_soft_limit=self.config.runtime_load_feeling.pending_index_soft_limit,
            family_overflow_soft_limit=self.config.runtime_load_feeling.family_overflow_soft_limit,
            residual_mass_soft_limit=self.config.runtime_load_feeling.residual_mass_soft_limit,
            mismatch_weight=self.config.runtime_load_feeling.mismatch_weight,
            fatigue_decay=self.config.runtime_load_feeling.fatigue_decay,
            fatigue_step=self.config.runtime_load_feeling.fatigue_step,
            fatigue_gain=self.config.runtime_load_feeling.fatigue_gain,
            max_energy=self.config.runtime_load_feeling.max_energy,
        )
        self.runtime_budget_controller = RuntimeBudgetController(
            enabled=self.config.runtime_budget_controller.enabled,
            smoothing_alpha=self.config.runtime_budget_controller.smoothing_alpha,
            readout_min_multiplier=self.config.runtime_budget_controller.readout_min_multiplier,
            readout_max_multiplier=self.config.runtime_budget_controller.readout_max_multiplier,
            attention_candidate_min_multiplier=self.config.runtime_budget_controller.attention_candidate_min_multiplier,
            attention_candidate_max_multiplier=self.config.runtime_budget_controller.attention_candidate_max_multiplier,
            index_jobs_min_multiplier=self.config.runtime_budget_controller.index_jobs_min_multiplier,
            index_jobs_max_multiplier=self.config.runtime_budget_controller.index_jobs_max_multiplier,
            index_time_min_multiplier=self.config.runtime_budget_controller.index_time_min_multiplier,
            index_time_max_multiplier=self.config.runtime_budget_controller.index_time_max_multiplier,
            trace_detail_min_multiplier=self.config.runtime_budget_controller.trace_detail_min_multiplier,
            trace_detail_max_multiplier=self.config.runtime_budget_controller.trace_detail_max_multiplier,
            min_r_state_items_per_head=self.config.runtime_budget_controller.min_r_state_items_per_head,
            preserve_1024_query_floor=self.config.runtime_budget_controller.preserve_1024_query_floor,
            max_extra_index_jobs=self.config.runtime_budget_controller.max_extra_index_jobs,
        )
        self.time_feeling = TimeFeelingChannel(
            enabled=self.config.time_feeling.enabled,
            threshold=self.config.time_feeling.threshold,
            gain=self.config.time_feeling.gain,
            min_confidence=self.config.time_feeling.min_confidence,
            default_radius_ticks=self.config.time_feeling.default_radius_ticks,
            recall_gain=self.config.time_feeling.recall_gain,
            fatigue_decay=self.config.time_feeling.fatigue_decay,
            fatigue_step=self.config.time_feeling.fatigue_step,
            fatigue_gain=self.config.time_feeling.fatigue_gain,
            fatigue_max=self.config.time_feeling.fatigue_max,
            max_sources=self.config.time_feeling.max_sources,
        )
        self.rhythm = RhythmChannel(
            enabled=self.config.rhythm.enabled,
            window=self.config.rhythm.window,
            min_hits=self.config.rhythm.min_hits,
            min_period=self.config.rhythm.min_period,
            max_period=self.config.rhythm.max_period,
            period_sigma_scale=self.config.rhythm.period_sigma_scale,
            phase_sigma_scale=self.config.rhythm.phase_sigma_scale,
            pulse_threshold=self.config.rhythm.pulse_threshold,
            phase_threshold=self.config.rhythm.phase_threshold,
            fatigue_decay=self.config.rhythm.fatigue_decay,
            fatigue_step=self.config.rhythm.fatigue_step,
            fatigue_gain=self.config.rhythm.fatigue_gain,
            fatigue_max=self.config.rhythm.fatigue_max,
            salience_threshold=self.config.rhythm.salience_threshold,
        )
        self.expectation_pressure = ExpectationPressureChannel(
            enabled=self.config.expectation_pressure.enabled,
            min_activation=self.config.expectation_pressure.min_activation,
            expectation_decay=self.config.expectation_pressure.expectation_decay,
            pressure_decay=self.config.expectation_pressure.pressure_decay,
            satisfaction_decay=self.config.expectation_pressure.satisfaction_decay,
            expectation_gain=self.config.expectation_pressure.expectation_gain,
            pressure_gain=self.config.expectation_pressure.pressure_gain,
            satisfaction_gain=self.config.expectation_pressure.satisfaction_gain,
            residual_gain=self.config.expectation_pressure.residual_gain,
            feedback_gain=self.config.expectation_pressure.feedback_gain,
        )
        self.expectation_anchor_verifier = BAnchorExpectationVerifier(
            enabled=self.config.expectation_pressure.anchor_verifier_enabled,
            max_anchors=self.config.expectation_pressure.anchor_max_anchors,
            decay=self.config.expectation_pressure.anchor_decay,
            min_anchor_level=self.config.expectation_pressure.anchor_min_level,
            min_outcome_virtual=self.config.expectation_pressure.anchor_min_outcome_virtual,
            validation_gain=self.config.expectation_pressure.anchor_validation_gain,
            miss_gain=self.config.expectation_pressure.anchor_miss_gain,
        )
        self.emotion_modulator = EmotionModulator(
            cfs_gain=self.config.emotion.cfs_gain,
            rwd_pun_gain=self.config.emotion.rwd_pun_gain,
        )
        self.innate_engine = InnateCodingEngine(
            enabled=self.config.innate_rules.enabled,
            min_fire_strength=self.config.innate_rules.min_fire_strength,
            max_items_per_phase=self.config.innate_rules.max_items_per_phase,
            max_action_nodes_per_phase=self.config.innate_rules.max_action_nodes_per_phase,
            max_learning_events_per_phase=self.config.innate_rules.max_learning_events_per_phase,
            apply_emit_sa=self.config.innate_rules.apply_emit_sa,
            apply_action_nodes=self.config.innate_rules.apply_action_nodes,
            apply_emotion_deltas=self.config.innate_rules.apply_emotion_deltas,
        )
        self.innate_learning_router = InnateLearningEventRouter()
        self.learning_event_builder = LearningEventBuilder()
        self.safety_gate = SafetyGate(
            enabled=self.config.innate_rules.safety_gate_enabled,
            veto_pressure_threshold=self.config.innate_rules.safety_veto_pressure_threshold,
            veto_cor_threshold=self.config.innate_rules.safety_veto_cor_threshold,
            review_pressure_threshold=self.config.innate_rules.safety_review_pressure_threshold,
            review_cor_threshold=self.config.innate_rules.safety_review_cor_threshold,
            min_external_confidence=self.config.innate_rules.safety_min_external_confidence,
        )
        self.adaptive_tuner = AdaptiveTuner(
            enabled=self.config.tuner.enabled,
            ema_alpha=self.config.tuner.ema_alpha,
            min_support_ticks=self.config.tuner.min_support_ticks,
            target_prediction_alignment=self.config.tuner.target_prediction_alignment,
            max_normal_pressure=self.config.tuner.max_normal_pressure,
            target_action_success=self.config.tuner.target_action_success,
            adjustment_rate=self.config.tuner.adjustment_rate,
            rollback_threshold=self.config.tuner.rollback_threshold,
        )
        self.action_planner = ActionConsequencePlanner(
            enabled=self.config.action.enabled,
            selection_threshold=self.config.action.selection_threshold,
            max_selected_actions=self.config.action.max_selected_actions,
            fatigue_decay=self.config.action.fatigue_decay,
            fatigue_step=self.config.action.fatigue_step,
            bias_learning_rate=self.config.action.bias_learning_rate,
            bias_gain=self.config.action.bias_gain,
            confidence_gain=self.config.action.confidence_gain,
            wait_base_drive=self.config.action.wait_base_drive,
            outcome_memory_enabled=self.config.action.outcome_memory_enabled,
            outcome_memory_learning_rate=self.config.action.outcome_memory_learning_rate,
            outcome_memory_decay_per_tick=self.config.action.outcome_memory_decay_per_tick,
            outcome_memory_support_scale=self.config.action.outcome_memory_support_scale,
            outcome_memory_max_drive_bias=self.config.action.outcome_memory_max_drive_bias,
        )
        self.action_consequence_evaluator = ActionConsequenceEvaluator(
            max_successor_rows=self.config.action.consequence_max_successor_rows,
            max_evidence_per_action=self.config.action.consequence_max_evidence_per_action,
            max_horizon=self.config.action.consequence_max_horizon,
            branching=self.config.action.consequence_branching,
            path_decay=self.config.action.consequence_path_decay,
        )
        self.action_control_effect_router = ActionControlEffectRouter()
        self.education_interventions = EducationInterventionBuffer()
        self.focus_buffer = FocusBuffer(
            focus_history_limit=self.config.short_term.focus_history_limit,
            recency_decay=self.config.short_term.recency_decay,
            synthetic_query_weight=self.config.short_term.synthetic_query_weight,
            replay_decay=self.config.short_term.replay_decay,
            replay_query_weight=self.config.short_term.replay_query_weight,
            max_replay_items=self.config.short_term.max_replay_items,
            episode_break_overlap=self.config.short_term.episode_break_overlap,
        )
        self.short_term_echo = ShortTermEchoBuffer(
            history_limit=self.config.short_term.echo_history_limit,
            max_age_ticks=self.config.short_term.echo_max_age_ticks,
            decay=self.config.short_term.echo_decay,
            sensory_gain=self.config.short_term.echo_sensory_gain,
            thought_gain=self.config.short_term.echo_thought_gain,
            max_echo_energy=self.config.short_term.echo_max_energy,
            max_items_per_tick=self.config.short_term.echo_max_items_per_tick,
            modality_policies={
                "vision": {
                    "max_age_ticks": self.config.short_term.echo_vision_max_age_ticks,
                    "decay": self.config.short_term.echo_vision_decay,
                    "sensory_gain": self.config.short_term.echo_vision_gain,
                    "max_energy": self.config.short_term.echo_vision_max_energy,
                },
                "audio": {
                    "max_age_ticks": self.config.short_term.echo_audio_max_age_ticks,
                    "decay": self.config.short_term.echo_audio_decay,
                    "sensory_gain": self.config.short_term.echo_audio_gain,
                    "max_energy": self.config.short_term.echo_audio_max_energy,
                },
                "text": {
                    "max_age_ticks": self.config.short_term.echo_text_max_age_ticks,
                    "decay": self.config.short_term.echo_text_decay,
                    "sensory_gain": self.config.short_term.echo_text_gain,
                    "max_energy": self.config.short_term.echo_text_max_energy,
                },
                "thought": {
                    "max_age_ticks": self.config.short_term.echo_thought_max_age_ticks,
                    "decay": self.config.short_term.echo_thought_decay,
                    "sensory_gain": self.config.short_term.echo_thought_modality_gain,
                    "thought_gain": self.config.short_term.echo_thought_modality_gain,
                    "max_energy": self.config.short_term.echo_thought_max_energy,
                },
            },
        )
        self.short_term_memory = ShortTermMemoryWindow(
            history_limit=self.config.short_term.memory_window_history_limit,
            max_age_ticks=self.config.short_term.memory_window_max_age_ticks,
            recency_decay=self.config.short_term.memory_window_recency_decay,
            fatigue_decay=self.config.short_term.memory_window_fatigue_decay,
            fatigue_step=self.config.short_term.memory_window_fatigue_step,
            max_items_per_event=self.config.short_term.memory_window_max_items_per_event,
            default_recall_limit=self.config.short_term.memory_window_recall_limit,
        )
        self.short_term_slot = ShortTermSlotPacketBuilder(
            enabled=self.config.short_term_slot.enabled,
            capacity=self.config.short_term_slot.capacity,
            base_virtual_budget=self.config.short_term_slot.base_virtual_budget,
            item_real_fraction=self.config.short_term_slot.item_real_fraction,
            item_min_virtual=self.config.short_term_slot.item_min_virtual,
            item_max_virtual=self.config.short_term_slot.item_max_virtual,
            item_rank_decay=self.config.short_term_slot.item_rank_decay,
            item_order_decay=self.config.short_term_slot.item_order_decay,
            summary_ratio=self.config.short_term_slot.summary_ratio,
            order_ratio=self.config.short_term_slot.order_ratio,
            continuity_ratio=self.config.short_term_slot.continuity_ratio,
            rhythm_ratio=self.config.short_term_slot.rhythm_ratio,
            load_floor=self.config.short_term_slot.load_floor,
            continuity_gain=self.config.short_term_slot.continuity_gain,
            order_gain=self.config.short_term_slot.order_gain,
            rhythm_gain=self.config.short_term_slot.rhythm_gain,
            working_memory_fill_limit=self.config.short_term_slot.working_memory_fill_limit,
            focus_merge_limit=self.config.short_term_slot.focus_merge_limit,
        )
        self.focus_successor_bias = FocusSuccessorBias(
            enabled=self.config.attention.successor_bias_enabled,
            context_limit=self.config.attention.successor_bias_context_limit,
            max_successors_per_context=self.config.attention.successor_bias_max_successors_per_context,
            max_context_labels=self.config.attention.successor_bias_max_context_labels,
            max_order=self.config.attention.successor_bias_max_order,
            top_k=self.config.attention.successor_bias_top_k,
            per_tick_update_limit=self.config.attention.successor_bias_per_tick_update_limit,
            real_threshold=self.config.attention.successor_bias_real_threshold,
            decay=self.config.attention.successor_bias_decay,
            rescale_threshold=self.config.attention.successor_bias_rescale_threshold,
            rescale_factor=self.config.attention.successor_bias_rescale_factor,
            min_support=self.config.attention.successor_bias_min_support,
            gain=self.config.attention.successor_bias_gain,
            max_bias=self.config.attention.successor_bias_max,
            entropy_floor=self.config.attention.successor_bias_entropy_floor,
        )
        self.text_actuator = TextActionActuator()
        self.visual_gaze_actuator = VisualGazeActuator()
        self.auditory_band_actuator = AuditoryBandActuator()
        self._pending_action_feedback: dict | None = None
        self._queued_external_feedback: dict | None = None
        self._pending_innate_attention_biases: list[dict] = []
        self._pending_action_attention_controls: list[dict] = []
        self._pending_slow_query_hints: list[dict] = []
        self._pending_focus_family_modulation: dict = {}
        self._last_short_term_slot_trace: dict | None = None
        self._active_text_successor_cursor: dict | None = None
        self._focus_hold_count = 0
        self._text_ingest_cache: dict = {}
        self._pending_dialogue_turn: dict | None = None
        self._recent_dialogue_inputs: list[str] = []
        self._dialogue_turn_serial = 0
        self.tick_index = -1

    def queue_education_intervention(self, intervention: dict) -> dict:
        """Queue one external teaching intervention for the next tick.

        AP core does not own a concrete teaching skill. It only accepts a
        neutral packet of state hints, soft drive biases, and feedback so human,
        detachable-rule, and LLM teachers all use the same non-invasive doorway.
        """

        return self.education_interventions.queue(intervention, tick_index=self.tick_index + 1)

    def process_text_tick(
        self,
        text: str = "",
        *,
        memory_bootstrap: bool = False,
        trace_mode: str | None = None,
        education_interventions: dict | list[dict] | None = None,
    ) -> dict:
        return self.process_multimodal_tick(
            text=text,
            memory_bootstrap=memory_bootstrap,
            trace_mode=trace_mode,
            education_interventions=education_interventions,
        )

    def process_multimodal_tick(
        self,
        text: str = "",
        *,
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        memory_bootstrap: bool = False,
        trace_mode: str | None = None,
        education_interventions: dict | list[dict] | None = None,
    ) -> dict:
        trace_mode = self._normalize_trace_mode(trace_mode)
        stage_started_at = perf_counter()
        performance_stages: list[dict] = []

        def mark_stage(stage_name: str) -> None:
            nonlocal stage_started_at
            now = perf_counter()
            performance_stages.append({"stage": stage_name, "ms": round((now - stage_started_at) * 1000.0, 4)})
            stage_started_at = now

        self.tick_index += 1
        gc_was_enabled = gc.isenabled()
        if bool(self.config.observability.disable_gc_during_tick) and gc_was_enabled:
            gc.disable()
        try:
            return self._process_multimodal_tick_inner(
                text=text,
                image_bytes=image_bytes,
                audio_bytes=audio_bytes,
                memory_bootstrap=memory_bootstrap,
                trace_mode=trace_mode,
                education_interventions=education_interventions,
                performance_stages=performance_stages,
                mark_stage=mark_stage,
                stage_started_at_ref=lambda: stage_started_at,
            )
        finally:
            if bool(self.config.observability.disable_gc_during_tick) and gc_was_enabled:
                gc.enable()

    def queue_external_feedback(
        self,
        *,
        reward: float = 0.0,
        punishment: float = 0.0,
        correctness: float = 0.0,
        confidence: float = 1.0,
        source: str = "external_feedback",
        notes: list[str] | None = None,
        **details,
    ) -> dict:
        feedback = {
            "schema_id": "external_action_feedback/v1",
            "reward": round(max(0.0, float(reward or 0.0)), 4),
            "punishment": round(max(0.0, float(punishment or 0.0)), 4),
            "correctness": round(max(0.0, float(correctness or 0.0)), 4),
            "confidence": round(max(0.0, min(1.0, float(confidence or 0.0))), 4),
            "source": str(source or "external_feedback"),
            "notes": [str(note or "") for note in list(notes or []) if str(note or "")],
        }
        for key, value in dict(details or {}).items():
            clean_key = str(key or "")
            if not clean_key or clean_key in feedback:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                feedback[clean_key] = value
            elif isinstance(value, (list, tuple)):
                feedback[clean_key] = [
                    item
                    for item in list(value)[:16]
                    if isinstance(item, (str, int, float, bool)) or item is None
                ]
            elif isinstance(value, dict):
                feedback[clean_key] = {
                    str(k or ""): v
                    for k, v in list(value.items())[:16]
                    if str(k or "") and (isinstance(v, (str, int, float, bool)) or v is None)
                }
        self._queued_external_feedback = feedback
        return {"schema_id": "external_feedback_queue_trace/v1", "queued": True, "feedback": dict(feedback)}

    def _queue_inline_education_interventions(self, interventions: dict | list[dict] | None) -> None:
        """Normalize caller-supplied teacher packets into the one-tick queue.

        This keeps education optional and detachable: if no packet is supplied,
        AP runs exactly as itself. If a packet is supplied, it is still consumed
        as state hints / soft biases / feedback rather than as a privileged
        skill routine inside the runtime.
        """

        if not interventions:
            return
        rows = interventions if isinstance(interventions, list) else [interventions]
        for row in rows:
            if isinstance(row, dict):
                self.queue_education_intervention(row)

    def _process_multimodal_tick_inner(
        self,
        *,
        text: str,
        image_bytes: bytes | None,
        audio_bytes: bytes | None,
        memory_bootstrap: bool,
        trace_mode: str,
        education_interventions: dict | list[dict] | None,
        performance_stages: list[dict],
        mark_stage,
        stage_started_at_ref,
    ) -> dict:
        runtime_budget_trace = self.runtime_budget_controller.begin_tick(self.tick_index)
        self.state_pool.begin_tick(self.tick_index)
        self._queue_inline_education_interventions(education_interventions)
        input_packet, competition, external_items, multimodal_trace = self._ingest_multimodal(text=text, image_bytes=image_bytes, audio_bytes=audio_bytes)
        focus_turn_boundary_trace = {"schema_id": "focus_external_turn_boundary/v1", "applied": False, "reason": "no_new_external_text"}
        text_turn_boundary_trace = {"schema_id": "text_action_external_turn_boundary/v1", "applied": False, "reason": "no_new_external_text"}
        state_pool_turn_boundary_trace = {"schema_id": "state_pool_external_turn_boundary/v1", "applied": False, "reason": "no_new_external_text"}
        memory_recall_turn_boundary_trace = {"schema_id": "memory_recall_turn_boundary/v1", "applied": False, "reason": "no_new_external_text"}
        if str(input_packet.get("normalized_text", "") or "").strip():
            self._clear_active_text_successor_cursor(reason="new_external_text_turn")
            memory_recall_turn_boundary = getattr(self.memory, "mark_recall_turn_boundary", None)
            if callable(memory_recall_turn_boundary):
                memory_recall_turn_boundary_trace = memory_recall_turn_boundary(
                    tick_index=self.tick_index,
                    reason="new_external_text_turn",
                )
            state_pool_turn_boundary_trace = self.state_pool.mark_external_turn_boundary(
                tick_index=self.tick_index,
                reason="new_external_text_turn",
            )
            text_turn_boundary_trace = self.text_actuator.mark_external_turn_boundary(
                tick_index=self.tick_index,
                reason="new_external_text_turn",
            )
            focus_turn_boundary_trace = self.focus_buffer.mark_external_turn_boundary(
                tick_index=self.tick_index,
                reason="new_external_text_turn",
            )
        dialogue_turn_trace = self._refresh_dialogue_turn_state(input_packet=input_packet, external_items=external_items)
        dialogue_turn_trace["focus_turn_boundary"] = focus_turn_boundary_trace
        dialogue_turn_trace["text_turn_boundary"] = text_turn_boundary_trace
        dialogue_turn_trace["state_pool_turn_boundary"] = state_pool_turn_boundary_trace
        dialogue_turn_trace["memory_recall_turn_boundary"] = memory_recall_turn_boundary_trace
        mark_stage("ingest")
        self._apply_external_or_bootstrap(external_items, memory_bootstrap=memory_bootstrap)
        dialogue_turn_items = list(dialogue_turn_trace.get("items", []) or [])
        if dialogue_turn_items:
            self.state_pool.apply_external_items(dialogue_turn_items, tick_index=self.tick_index)
        early_text_context_items = self.text_actuator.short_term_context_items()
        if early_text_context_items:
            # The draft surface is part of AP's own perceivable environment:
            # after writing a character, the next tick should be able to recall
            # successors from "what I can see I already wrote" before action
            # planning. This is not an answer hint; it is self-observation of
            # low-grain text actions and cursor state.
            self.state_pool.apply_external_items(early_text_context_items, tick_index=self.tick_index)
        short_term_echo_trace = self._apply_short_term_echo(external_items)
        short_term_memory_observations: list[dict] = [
            self._observe_short_term_memory(external_items, source_kind="sensory", role="current_external_input")
        ]
        if dialogue_turn_items:
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    dialogue_turn_items,
                    source_kind="dialogue_turn_state",
                    modality="dialogue",
                    role="current_turn_closure_state",
                )
            )
        if early_text_context_items:
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    early_text_context_items,
                    source_kind="text_action",
                    modality="draft",
                    role="self_observed_draft_surface",
                )
            )
        education_intervention_trace = self.education_interventions.consume(tick_index=self.tick_index)
        education_state_items = list(education_intervention_trace.get("state_items", []) or [])
        if education_state_items:
            self.state_pool.apply_external_items(education_state_items, tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    education_state_items,
                    source_kind="education_intervention",
                    modality="teacher",
                    role="external_teacher_hint",
                )
            )
        education_feedback = dict(education_intervention_trace.get("feedback", {}) or {})
        if education_feedback:
            self._merge_queued_external_feedback(education_feedback)
        action_feedback_trace = self._consume_pending_action_feedback()
        if action_feedback_trace["feedback_items"]:
            self.state_pool.apply_external_items(action_feedback_trace["feedback_items"], tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    list(action_feedback_trace.get("feedback_items", []) or []),
                    source_kind="action_feedback",
                    modality="action",
                    role="delayed_action_feedback",
                )
            )
        mark_stage("apply_external")
        innate_traces: dict[str, dict] = {}
        post_external_innate_trace = self._evaluate_innate_phase(
            "post_external",
            state_items=external_items + list(action_feedback_trace.get("feedback_items", []) or []),
            action_feedback_trace=action_feedback_trace,
        )
        innate_traces["post_external"] = post_external_innate_trace
        mark_stage("innate_post_external")

        # Emotion is a slow modulation layer. The attention gate uses the
        # modulation state accumulated before this tick; current feelings and
        # feedback update the layer later and affect action + subsequent ticks.
        prior_emotion_modulation = self.emotion_modulator.get_modulation()

        readout_budget = self.runtime_budget_controller.readout_budget(
            base_items_per_head=self.config.state_pool.r_state_items_per_head,
            base_head_limit=self.config.state_pool.r_state_head_limit,
        )
        runtime_budget_trace["readout_budget"] = readout_budget
        r_state_fast = self.state_pool.read_r_state(
            items_per_head=readout_budget["items_per_head"],
            head_limit=readout_budget["head_limit"],
        )
        r_state_fast["runtime_budget"] = {
            "schema_id": "r_state_runtime_budget/v1",
            "items_per_head_changed": bool(readout_budget.get("changed", False)),
            "source_tick_index": int(runtime_budget_trace.get("source_tick_index", -1) or -1),
        }
        if education_state_items:
            # Current education/process-feeling items are already ordinary
            # state-pool citizens. Keep them visible to the bounded fast-query
            # readout for this tick so recent process feelings can compete
            # with older residual context without injecting an action answer.
            r_state_fast = self._append_r_state_head(
                r_state_fast,
                "head_education_intervention",
                education_state_items,
            )
        fast_query = self._r_state_to_query_items(r_state_fast)
        fast_bn, fast_cn = self._run_recall_branch(fast_query, memory_kind="state", prediction_source="fast_cn")
        mark_stage("fast_recall_initial")
        innate_traces["post_fast_recall"] = self._evaluate_innate_phase(
            "post_fast_recall",
            state_items=self._r_state_to_query_items(r_state_fast),
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            prediction_trace=self.state_pool.prediction_trace(),
            action_feedback_trace=action_feedback_trace,
        )
        mark_stage("innate_post_fast_recall")
        time_trace = self.time_feeling.derive(tick_index=self.tick_index, bn_rows=fast_bn)
        time_context = self._build_time_context(time_trace)
        if time_trace["items"]:
            self.state_pool.apply_external_items(time_trace["items"], tick_index=self.tick_index)
            time_rows = self.state_pool.rows_for_labels(
                [str(item.get("sa_label", "") or "") for item in time_trace["items"] if str(item.get("sa_label", "") or "")]
            )
            if time_rows:
                r_state_fast = self._append_r_state_head(r_state_fast, "head_time_feeling", time_rows)
            if self._should_rerun_timefelt_recall(time_trace):
                fast_query = self._r_state_to_query_items(r_state_fast)
                fast_bn, fast_cn = self._run_recall_branch(
                    fast_query,
                    memory_kind="state",
                    prediction_source="fast_cn_timefelt",
                    time_context=time_context,
                )
        mark_stage("time_and_fast_recall")

        previous_focus_labels = self.focus_buffer.tail()
        innate_attention_biases_for_selection = self._consume_pending_innate_attention_biases()
        action_attention_controls_for_selection = self._consume_pending_action_attention_controls()
        action_focus_family_modulation = self._consume_pending_focus_family_modulation()
        attention_candidates = self._r_state_to_attention_candidates(r_state_fast)
        action_attention_controls_for_selection = self._enrich_action_attention_controls_with_learned_bands(
            action_attention_controls_for_selection,
            attention_candidates=attention_candidates,
        )
        successor_bias_trace = self.focus_successor_bias.build_bias(
            previous_focus_labels=previous_focus_labels,
            candidate_items=attention_candidates,
            tick_index=self.tick_index,
        )
        attention_trace = self.attention.select(
            attention_candidates,
            previous_focus_labels=previous_focus_labels,
            emotion_modulation=prior_emotion_modulation,
            successor_bias=successor_bias_trace,
            innate_attention_biases=innate_attention_biases_for_selection,
            action_attention_controls=action_attention_controls_for_selection,
        )
        raw_attention_items = list(attention_trace.get("selected_items", []) or [])
        balanced_focus_items, focus_family_budget_trace = self._shape_focus_family_budget(
            ranked_items=list(attention_trace.get("ranked_items", []) or []),
            raw_selected_items=raw_attention_items,
            action_modulation=action_focus_family_modulation,
        )
        ordered_focus_items, focus_order_trace = self._stabilize_focus_order(balanced_focus_items)
        ordered_focus_labels = [
            str(item.get("sa_label", "") or "")
            for item in ordered_focus_items
            if str(item.get("sa_label", "") or "")
        ]
        attention_trace = dict(attention_trace)
        attention_trace["raw_selected_labels"] = [
            str(item.get("sa_label", "") or "")
            for item in raw_attention_items
            if str(item.get("sa_label", "") or "")
        ]
        attention_trace["family_budget"] = focus_family_budget_trace
        attention_trace["action_attention_controls_consumed"] = action_attention_controls_for_selection
        attention_trace["action_family_budget_modulation_consumed"] = action_focus_family_modulation
        attention_trace["selected_items"] = ordered_focus_items
        attention_trace["selected_labels"] = ordered_focus_labels
        attention_trace["focus_order"] = focus_order_trace
        self.state_pool.select_focus(ordered_focus_labels)
        self.focus_buffer.push(ordered_focus_items, tick_index=self.tick_index)
        self._observe_short_term_thought_echo(ordered_focus_items)
        short_term_memory_observations.append(
            self._observe_short_term_memory(ordered_focus_items, source_kind="thought", modality="thought", role="selected_attention_focus")
        )
        successor_bias_update_trace = self.focus_successor_bias.observe_transition(
            previous_focus_labels=previous_focus_labels,
            current_focus_items=ordered_focus_items,
            tick_index=self.tick_index,
        )
        focus_continuation_trace = self.focus_buffer.trace(tick_index=self.tick_index)
        short_term_preview_recall = self.short_term_memory.recall(
            tick_index=self.tick_index,
            cues=ordered_focus_items,
            limit=self.config.short_term.memory_window_recall_limit,
            reason="planning_preview",
            similarity_fn=self._short_term_memory_similarity,
            update_fatigue=False,
        ) if bool(getattr(self.config.short_term, "memory_window_enabled", True)) else {"available": False}
        focus_continuation_trace["short_term_memory_readback"] = short_term_preview_recall
        self.rhythm.observe(tick_index=self.tick_index, focus_items=ordered_focus_items)
        rhythm_trace = self.rhythm.derive(tick_index=self.tick_index)
        if rhythm_trace["items"]:
            self.state_pool.apply_external_items(rhythm_trace["items"], tick_index=self.tick_index)
        runtime_load_for_slot = {
            "schema_id": "runtime_load_slot_view/v1",
            "tick_index": self.tick_index,
            "channels": {
                "load_ratio": float(
                    min(
                        1.0,
                        (
                            len(attention_candidates) / max(1, self.config.runtime_load_feeling.attention_candidate_soft_limit)
                            + len(self.state_pool.snapshot().get("items", []) or []) / max(1, self.config.runtime_load_feeling.state_item_soft_limit)
                        )
                        * 0.5,
                    )
                ),
            },
            "state_snapshot": self.state_pool.snapshot(),
            "attention_trace": {
                "selected_count": len(ordered_focus_items),
                "candidate_count": len(attention_candidates),
            },
            "prediction_trace": self.state_pool.prediction_trace(),
            "residual_summary": self.state_pool.residual_summary(limit=8),
            "pending_index_summary": self.memory.pending_index_job_summary(),
        }
        short_term_slot_trace = self.short_term_slot.build(
            tick_index=self.tick_index,
            focus_items=ordered_focus_items,
            focus_continuation_trace=focus_continuation_trace,
            short_term_memory_trace=self.short_term_memory.trace(tick_index=self.tick_index, last_recall=short_term_preview_recall),
            rhythm_trace=rhythm_trace,
            runtime_load_trace=runtime_load_for_slot,
            state_rows=self.state_pool.rows_for_labels([str(item.get("sa_label", "") or "") for item in ordered_focus_items if isinstance(item, dict)]),
        )
        self._last_short_term_slot_trace = dict(short_term_slot_trace)
        if short_term_slot_trace.get("items"):
            self.state_pool.apply_external_items(list(short_term_slot_trace.get("items", []) or []), tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    list(short_term_slot_trace.get("items", []) or []),
                    source_kind="short_term_slot",
                    modality="thought",
                    role="slot_packet",
                )
            )
        mark_stage("attention_and_rhythm")
        innate_traces["post_attention"] = self._evaluate_innate_phase(
            "post_attention",
            state_items=ordered_focus_items,
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            attention=attention_trace,
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            action_feedback_trace=action_feedback_trace,
        )
        mark_stage("innate_post_attention")

        slow_query_hints_for_selection = self._consume_pending_slow_query_hints()
        focus_continuation_trace["action_slow_query_hints_consumed"] = slow_query_hints_for_selection
        slow_query = self._build_slow_query(ordered_focus_items, action_slow_query_hints=slow_query_hints_for_selection)
        slow_bn, slow_cn = self._run_recall_branch(
            slow_query,
            memory_kind="focus",
            prediction_source="slow_cn",
            time_context=time_context,
        )
        active_text_successor_trace = self._read_active_text_successor_branch()
        if active_text_successor_trace.get("rows"):
            active_rows = [dict(row) for row in list(active_text_successor_trace.get("rows", []) or []) if isinstance(row, dict)]
            slow_cn = self._merge_successor_branch_rows(active_rows, slow_cn)
            active_predicted_items = [
                item
                for branch in active_rows
                for item in list(branch.get("predicted_items", []) or [])
                if isinstance(item, dict)
            ]
            if active_predicted_items:
                self.state_pool.apply_predictions(
                    active_predicted_items,
                    tick_index=self.tick_index,
                    source="active_text_successor_branch",
                )
        focus_continuation_trace["active_text_successor_branch"] = active_text_successor_trace
        mark_stage("slow_recall")
        innate_traces["post_slow_recall"] = self._evaluate_innate_phase(
            "post_slow_recall",
            state_items=ordered_focus_items,
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            attention=attention_trace,
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            action_feedback_trace=action_feedback_trace,
        )
        mark_stage("innate_post_slow_recall")

        state_snapshot_before_feelings = self.state_pool.snapshot()
        prediction_trace_for_feelings = self.state_pool.prediction_trace()
        residual_summary_for_feelings = self.state_pool.residual_summary(limit=8)
        feeling_trace = self.cognitive_feelings.derive(
            state_snapshot_items=state_snapshot_before_feelings["items"],
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            prediction_trace=prediction_trace_for_feelings,
            residual_summary=residual_summary_for_feelings,
        )
        post_validation_innate_trace = self._evaluate_innate_phase(
            "post_prediction_validation",
            state_items=state_snapshot_before_feelings["items"],
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            attention=attention_trace,
            feelings=feeling_trace,
            prediction_trace=prediction_trace_for_feelings,
            residual_summary=residual_summary_for_feelings,
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            action_feedback_trace=action_feedback_trace,
        )
        innate_traces["post_prediction_validation"] = post_validation_innate_trace
        if feeling_trace["items"]:
            self.state_pool.apply_external_items(feeling_trace["items"], tick_index=self.tick_index)
        if post_validation_innate_trace["items"]:
            self.state_pool.apply_external_items(post_validation_innate_trace["items"], tick_index=self.tick_index)
        mark_stage("cognitive_feelings")

        # P1-J-17 task feelings must be part of the same cognitive tick, not a
        # post-hoc log row. We derive them from the current successor/readback
        # situation before expectation pressure and emotion update, so boredom
        # and fulfillment can modulate NT state and action planning immediately.
        short_term_memory_trace = self._short_term_memory_trace(last_recall=short_term_preview_recall)
        short_term_memory_trace["observations"] = [dict(row) for row in short_term_memory_observations if isinstance(row, dict)]
        task_feeling_trace = self._derive_task_feeling(
            input_packet=input_packet,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            focus_continuation_trace=focus_continuation_trace,
            short_term_memory_trace=short_term_memory_trace,
            cognitive_feelings=feeling_trace,
            residual_summary=self.state_pool.residual_summary(limit=8),
            action_trace={},
        )
        if task_feeling_trace.get("items"):
            self.state_pool.apply_external_items(list(task_feeling_trace.get("items", []) or []), tick_index=self.tick_index)
            feeling_trace = self._merge_feelings_with_task_feeling(feeling_trace, task_feeling_trace)
        mark_stage("task_feeling")

        expectation_pressure_trace = self.expectation_pressure.derive(
            tick_index=self.tick_index,
            cognitive_feelings=feeling_trace,
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            action_feedback_trace=action_feedback_trace,
            rhythm_trace=rhythm_trace,
            time_trace=time_trace,
        )
        if expectation_pressure_trace["items"]:
            self.state_pool.apply_external_items(expectation_pressure_trace["items"], tick_index=self.tick_index)
        expectation_anchor_trace = self.expectation_anchor_verifier.update(
            tick_index=self.tick_index,
            fast_bn=fast_bn,
            slow_bn=slow_bn,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            action_feedback_trace=action_feedback_trace,
            cognitive_feelings=feeling_trace,
        )
        expectation_pressure_trace = dict(expectation_pressure_trace)
        expectation_pressure_trace["anchor_verification"] = expectation_anchor_trace
        expectation_pressure_trace["items"] = list(expectation_pressure_trace.get("items", []) or []) + list(expectation_anchor_trace.get("items", []) or [])
        channels = dict(expectation_pressure_trace.get("channels", {}) or {})
        channels["anchor_active_count"] = int(expectation_anchor_trace.get("active_count", 0) or 0)
        channels["anchor_created_count"] = len(expectation_anchor_trace.get("created", []) or [])
        channels["anchor_verified_count"] = len(expectation_anchor_trace.get("verified", []) or [])
        channels["anchor_missed_count"] = len(expectation_anchor_trace.get("missed", []) or [])
        expectation_pressure_trace["channels"] = channels
        if expectation_anchor_trace.get("items"):
            self.state_pool.apply_external_items(list(expectation_anchor_trace.get("items", []) or []), tick_index=self.tick_index)
        mark_stage("expectation_pressure")

        observed_feedback_for_emotion = dict(action_feedback_trace.get("observed_feedback", {}) or {})
        emotion_post_innate_trace = self._evaluate_innate_phase(
            "emotion_post",
            state_items=self.state_pool.snapshot()["items"],
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            attention=attention_trace,
            feelings=feeling_trace,
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            expectation_pressure=expectation_pressure_trace,
            emotion_state=self.emotion_modulator.state.get_state(),
            action_feedback_trace=action_feedback_trace,
        )
        innate_traces["emotion_post"] = emotion_post_innate_trace
        # Update emotion state based on cognitive feelings and delayed action feedback
        # (8-channel NT system). This post-attention update is available to action
        # planning immediately and to attention on the next tick.
        emotion_feelings = self._merge_feelings_with_expectation_pressure(
            feeling_trace,
            expectation_pressure_trace,
        )
        emotion_update_trace = self.emotion_modulator.update(
            cognitive_feelings=emotion_feelings,
            reward=float(observed_feedback_for_emotion.get("reward", 0.0) or 0.0),
            punishment=float(observed_feedback_for_emotion.get("punishment", 0.0) or 0.0),
            innate_deltas=emotion_post_innate_trace.get("emotion_deltas", {}),
        )
        emotion_modulation = self.emotion_modulator.get_modulation()

        state_snapshot_before_action = self.state_pool.snapshot()
        current_draft_state = self.text_actuator.draft_state(current_tick=self.tick_index)
        text_context_items = self.text_actuator.short_term_context_items()
        if current_draft_state:
            text_context_items = [
                {
                    "sa_label": "text_action::draft_state",
                    "display_text": f"draft:{str(current_draft_state.get('visible_text', '') or '')}",
                    "family": "text_action",
                    "source_type": "text_action",
                    "real_energy": 0.12,
                    "virtual_energy": 0.02,
                    "cognitive_pressure": 0.04,
                    "anchor_meta": dict(current_draft_state),
                }
            ] + list(text_context_items or [])
        if text_context_items:
            # Draft text actions are low-energy, local actuator facts. They can
            # fall out of the global state snapshot when prediction/residual
            # items are loud, but the drive manager still needs them to decide
            # whether a recent output mismatch should be reread or revised.
            # We feed them as planning context only; normal output items are
            # still written to the state pool after the actuator acts. The
            # active draft_state row is always current readback, even when it is
            # empty, so old text_action memories cannot masquerade as text that
            # is still visible on the draft surface.
            seen_snapshot_labels = {str(item.get("sa_label", "") or "") for item in state_snapshot_before_action.get("items", [])}
            merged_items = list(state_snapshot_before_action.get("items", []) or [])
            if current_draft_state:
                merged_items = [
                    item
                    for item in merged_items
                    if not (
                        str(item.get("sa_label", "") or "") == "text_action::draft_state"
                        or str((item.get("anchor_meta", {}) or {}).get("schema_id", "") or "") == "text_draft_state/v1"
                    )
                ]
                seen_snapshot_labels = {str(item.get("sa_label", "") or "") for item in merged_items}
            merged_items.extend(
                item
                for item in text_context_items
                if str(item.get("sa_label", "") or "") not in seen_snapshot_labels
                or str((item.get("anchor_meta", {}) or {}).get("event_type", "") or "") in {"write_mismatch", "reread", "revise", "replace"}
            )
            state_snapshot_before_action = dict(state_snapshot_before_action)
            state_snapshot_before_action["items"] = merged_items
        if education_state_items:
            # External teacher hints are first-class state-field citizens. They
            # make the current teaching context recallable, but teacher control
            # still remains a soft drive bias consumed by the planner below.
            seen_snapshot_labels = {str(item.get("sa_label", "") or "") for item in state_snapshot_before_action.get("items", [])}
            extra_education_items = [
                item
                for item in education_state_items
                if str(item.get("sa_label", "") or "") and str(item.get("sa_label", "") or "") not in seen_snapshot_labels
            ]
            if extra_education_items:
                state_snapshot_before_action = dict(state_snapshot_before_action)
                state_snapshot_before_action["items"] = list(state_snapshot_before_action.get("items", []) or []) + extra_education_items
        if dialogue_turn_items:
            # The active user-turn closure state is a short-lived process
            # anchor. It must reach action competition even when broad memory
            # residue crowds it out of a bounded state-pool snapshot.
            seen_snapshot_labels = {str(item.get("sa_label", "") or "") for item in state_snapshot_before_action.get("items", [])}
            extra_dialogue_items = [
                item
                for item in dialogue_turn_items
                if str(item.get("sa_label", "") or "") and str(item.get("sa_label", "") or "") not in seen_snapshot_labels
            ]
            if extra_dialogue_items:
                state_snapshot_before_action = dict(state_snapshot_before_action)
                state_snapshot_before_action["items"] = list(state_snapshot_before_action.get("items", []) or []) + extra_dialogue_items
        commit_readiness_context = self.action_planner.draft_commit_readiness_context(
            state_snapshot_before_action["items"],
            current_tick=self.tick_index,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            correctness=float(feeling_trace.get("channels", {}).get("correctness", 0.0) or 0.0),
            grasp=float(feeling_trace.get("channels", {}).get("grasp", 0.0) or 0.0),
            pressure=float(feeling_trace.get("channels", {}).get("pressure", 0.0) or 0.0),
            dissonance=float(feeling_trace.get("channels", {}).get("dissonance", 0.0) or 0.0),
            uncertainty=float(feeling_trace.get("channels", {}).get("uncertainty", 0.0) or 0.0),
            pressure_anchor_level=float(expectation_pressure_trace.get("channels", {}).get("pressure_level", 0.0) or 0.0) if expectation_pressure_trace else 0.0,
            expectation_gap=float(expectation_pressure_trace.get("channels", {}).get("expectation_gap", 0.0) or 0.0) if expectation_pressure_trace else 0.0,
        )
        if commit_readiness_context:
            commit_readiness_item = {
                "sa_label": "state::commit_ready",
                "display_text": "commit_ready",
                "family": "state",
                "source_type": "text_action",
                "real_energy": float(commit_readiness_context.get("commit_readiness", 0.0) or 0.0),
                "virtual_energy": float(commit_readiness_context.get("commit_reread_need", 0.0) or 0.0) * 0.42,
                "cognitive_pressure": float(commit_readiness_context.get("commit_readiness", 0.0) or 0.0),
                "anchor_meta": {
                    **dict(commit_readiness_context),
                    "schema_id": "text_commit_readiness_state/v1",
                    "tick_index": int(self.tick_index),
                    "policy": "short_lived_commit_readiness_is_learnable_state_not_force_submit",
                },
            }
            self.state_pool.apply_external_items([commit_readiness_item], tick_index=self.tick_index)
            state_snapshot_before_action = dict(state_snapshot_before_action)
            state_snapshot_before_action["items"] = list(state_snapshot_before_action.get("items", []) or []) + [commit_readiness_item]
        action_consequence_trace = self.action_consequence_evaluator.evaluate(
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            snapshot_lookup=self.memory.snapshot_by_id,
            successor_lookup=self.memory.successor_links,
            current_tick=self.tick_index,
        )
        mark_stage("emotion_and_consequence")
        draft_context_before_action = self.text_actuator.draft_state(current_tick=self.tick_index)
        action_preselect_innate_trace = self._evaluate_innate_phase(
            "action_preselect",
            state_items=state_snapshot_before_action["items"],
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            attention=attention_trace,
            feelings=feeling_trace,
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            expectation_pressure=expectation_pressure_trace,
            emotion_state=emotion_update_trace.get("emotion_state", {}),
            action_feedback_trace=action_feedback_trace,
            action_consequence_trace=action_consequence_trace,
            draft_context=draft_context_before_action,
        )
        innate_traces["action_preselect"] = action_preselect_innate_trace
        action_biases_for_planner = list(action_preselect_innate_trace.get("action_biases", []) or []) + list(
            education_intervention_trace.get("action_biases", []) or []
        )
        action_trace = self.action_planner.plan(
            tick_index=self.tick_index,
            state_snapshot_items=state_snapshot_before_action["items"],
            attention_trace=attention_trace,
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            cognitive_feelings=feeling_trace,
            expectation_pressure_trace=expectation_pressure_trace,
            rhythm_trace=rhythm_trace,
            time_trace=time_trace,
            residual_summary=self.state_pool.residual_summary(limit=8),
            prediction_trace=self.state_pool.prediction_trace(),
            action_consequence_trace=action_consequence_trace,
            emotion_modulation=emotion_modulation,
            innate_action_nodes=action_preselect_innate_trace.get("action_nodes", []),
            innate_action_biases=action_biases_for_planner,
            recent_thought_readback=dict(focus_continuation_trace.get("recent_thought_readback", {}) or {}),
            short_term_memory_readback=dict(focus_continuation_trace.get("short_term_memory_readback", {}) or {}),
            memory_action_drive_gain=self.config.innate_rules.memory_action_virtual_drive_gain,
        )
        action_trace["education_intervention"] = education_intervention_trace
        pre_safety_short_term_recall = self._short_term_memory_recall_for_actions(
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            attention_trace=attention_trace,
            state_snapshot_items=state_snapshot_before_action.get("items", []),
            focus_continuation_trace=focus_continuation_trace,
            reason="pre_safety_action_control",
            update_fatigue=False,
        )
        pre_safety_control_items = self._build_action_control_items(
            selected_actions=action_trace.get("selected_actions", []),
            attention_trace=attention_trace,
            fast_bn=fast_bn,
            slow_bn=slow_bn,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            state_snapshot_items=state_snapshot_before_action.get("items", []),
            time_context=time_context,
            action_consequence_trace=action_consequence_trace,
            expectation_pressure_trace=expectation_pressure_trace,
            focus_continuation_trace=focus_continuation_trace,
            short_term_memory_recall=pre_safety_short_term_recall,
        )
        safety_gate_trace = self.safety_gate.review(
            tick_index=self.tick_index,
            candidates=list(action_trace.get("candidates", []) or []),
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            cognitive_feelings=feeling_trace,
            emotion_state=emotion_update_trace.get("emotion_state", {}),
            safety_trace={"hits": list(action_preselect_innate_trace.get("safety_gate", []) or [])},
            expectation_pressure_trace=expectation_pressure_trace,
            action_control_items=pre_safety_control_items,
        )
        if safety_gate_trace.get("enabled", False):
            action_trace["selected_actions_before_safety"] = list(action_trace.get("selected_actions", []) or [])
            action_trace["selected_actions"] = list(safety_gate_trace.get("selected_actions", []) or [])
            action_trace["action_items"] = self.action_planner.build_action_items(
                list(action_trace.get("selected_actions", []) or []),
                tick_index=self.tick_index,
            )
        action_trace["safety_gate"] = safety_gate_trace
        action_trace["feedback_items"] = list(action_feedback_trace.get("feedback_items", []) or [])
        selected_short_term_recall = self._short_term_memory_recall_for_actions(
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            attention_trace=attention_trace,
            state_snapshot_items=state_snapshot_before_action.get("items", []),
            focus_continuation_trace=focus_continuation_trace,
            reason="selected_action_control",
            update_fatigue=True,
        )
        control_items = self._build_action_control_items(
            selected_actions=action_trace.get("selected_actions", []),
            attention_trace=attention_trace,
            fast_bn=fast_bn,
            slow_bn=slow_bn,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            state_snapshot_items=state_snapshot_before_action.get("items", []),
            time_context=time_context,
            action_consequence_trace=action_consequence_trace,
            expectation_pressure_trace=expectation_pressure_trace,
            focus_continuation_trace=focus_continuation_trace,
            short_term_memory_recall=selected_short_term_recall,
        )
        action_control_effect_trace = self.action_control_effect_router.build(
            tick_index=self.tick_index,
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            attention_trace=attention_trace,
            state_snapshot_items=state_snapshot_before_action.get("items", []),
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            previous_focus_labels=previous_focus_labels,
        )
        effect_control_items = list(action_control_effect_trace.get("control_items", []) or [])
        if effect_control_items:
            control_items = list(control_items) + effect_control_items
        self._remember_action_control_effects(action_control_effect_trace)
        visual_control_trace = self.visual_gaze_actuator.step(
            tick_index=self.tick_index,
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            attention_trace=attention_trace,
        )
        auditory_control_trace = self.auditory_band_actuator.step(
            tick_index=self.tick_index,
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            attention_trace=attention_trace,
        )
        control_items = (
            list(control_items)
            + list(visual_control_trace.get("items", []) or [])
            + list(auditory_control_trace.get("items", []) or [])
        )
        if control_items:
            self.state_pool.apply_external_items(control_items, tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(control_items, source_kind="action_control", modality="action", role="selected_action_controls")
            )
        if action_trace["action_items"]:
            self.state_pool.apply_external_items(action_trace["action_items"], tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    list(action_trace.get("action_items", []) or []),
                    source_kind="action",
                    modality="action",
                    role="selected_action_nodes",
                )
            )
        if safety_gate_trace.get("inhibition_items"):
            self.state_pool.apply_external_items(list(safety_gate_trace.get("inhibition_items", []) or []), tick_index=self.tick_index)
        action_trace["control_items"] = control_items
        action_trace["action_control_effects"] = action_control_effect_trace
        action_trace["visual_gaze"] = visual_control_trace
        action_trace["auditory_band"] = auditory_control_trace
        action_trace["inhibition_items"] = list(safety_gate_trace.get("inhibition_items", []) or [])
        action_trace["short_term_memory_recall"] = selected_short_term_recall
        unfinished_mark_trace = self._mark_unfinished_thought_if_needed(
            focus_continuation_trace=focus_continuation_trace,
            expected_text=self.action_planner.expected_text_context(
                fast_bn=fast_bn,
                slow_bn=slow_bn,
                fast_cn=fast_cn,
                slow_cn=slow_cn,
                draft_context=self.text_actuator.draft_state(current_tick=self.tick_index),
            ),
            action_trace=action_trace,
            short_term_memory_recall=selected_short_term_recall,
        )
        action_trace["unfinished_thought_mark"] = unfinished_mark_trace
        feedback_focus_rows = attention_trace.get("selected_items", [])[: self.config.attention.focus_limit]
        top_after_control = self._labels_after_action_control(
            feedback_focus_rows=feedback_focus_rows,
            control_items=control_items,
            action_items=list(action_trace.get("action_items", []) or []),
        )
        action_trace["feedback_context"] = {
            "focus_labels_after_control": [str(item.get("sa_label", "") or "") for item in feedback_focus_rows],
            "top_labels_after_control": top_after_control,
            "visual_gaze_events": [dict(event) for event in list(visual_control_trace.get("events", []) or [])],
            "auditory_band_events": [dict(event) for event in list(auditory_control_trace.get("events", []) or [])],
            "action_control_effects": [
                dict((item.get("anchor_meta", {}) or {}))
                for item in list(control_items or [])[:16]
                if str((item.get("anchor_meta", {}) or {}).get("schema_id", "") or "").endswith("_control/v1")
            ],
        }

        # Text output-side closure: explicit write/reread/revise evidence chain.
        # This is not "the model output"; it is a white-box actuator trace that
        # can be audited and learned from later.
        focus_labels_now = list(attention_trace.get("selected_labels", []) or [])
        text_output_trace = self.text_actuator.step(
            tick_index=self.tick_index,
            input_text=input_packet.get("normalized_text", "") or "",
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            focus_labels=focus_labels_now,
            cognitive_feelings=feeling_trace,
        )
        dialogue_turn_close_trace = self._close_dialogue_turn_if_committed(text_output_trace)
        text_output_trace = dict(text_output_trace)
        text_output_trace["dialogue_turn_close"] = dialogue_turn_close_trace
        text_successor_cursor_update = self._update_active_text_successor_cursor(
            text_output_trace=text_output_trace,
            fast_cn=fast_cn,
            slow_cn=slow_cn,
        )
        text_output_trace["active_successor_cursor_update"] = text_successor_cursor_update
        dialogue_turn_trace["close"] = dialogue_turn_close_trace
        if text_output_trace.get("output_items"):
            self.state_pool.apply_external_items(list(text_output_trace.get("output_items", []) or []), tick_index=self.tick_index)
            short_term_memory_observations.append(
                self._observe_short_term_memory(
                    list(text_output_trace.get("output_items", []) or []),
                    source_kind="output",
                    modality="text",
                    role="text_action_output",
                )
            )
        short_term_memory_trace = self._short_term_memory_trace(last_recall=selected_short_term_recall)
        short_term_memory_trace["observations"] = [dict(row) for row in short_term_memory_observations if isinstance(row, dict)]
        short_term_memory_trace["slot_packet"] = short_term_slot_trace
        task_feeling_trace = dict(task_feeling_trace)
        task_feeling_trace["post_action_short_term_memory"] = {
            "schema_id": "task_feeling_post_action_memory_note/v1",
            "last_recall_available": bool(selected_short_term_recall.get("available", False)),
            "selected_action_count": len(list(action_trace.get("selected_actions", []) or [])),
            "meaning": "task_feeling_was_derived_before_emotion;this_note_keeps_late_memory_recall_observable",
        }
        mark_stage("action_and_output")

        state_snapshot = self.state_pool.snapshot()
        state_snapshot_for_memory = self.state_pool.snapshot_for_memory_write()
        state_snapshot["energy_flow"] = self.state_pool.energy_flow_trace(
            items=state_snapshot.get("items", []),
            r_state=r_state_fast,
            memory_write_items=state_snapshot_for_memory.get("items", []),
        )
        target_tick_ms = float(getattr(self.config.observability, "target_tick_ms", 100) or 100)
        pending_index_before_write = self.memory.pending_index_job_summary()
        elapsed_before_load_feeling_ms = (perf_counter() - stage_started_at_ref()) * 1000.0 + sum(
            float(stage.get("ms", 0.0) or 0.0)
            for stage in performance_stages
        )
        runtime_load_trace = self.runtime_load_feeling.derive(
            tick_index=self.tick_index,
            target_tick_ms=target_tick_ms,
            elapsed_ms=elapsed_before_load_feeling_ms,
            r_state=r_state_fast,
            attention_candidates=attention_candidates,
            state_snapshot=state_snapshot,
            attention_trace=attention_trace,
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            pending_index_summary=pending_index_before_write,
        )
        if runtime_load_trace["items"]:
            self.state_pool.apply_external_items(runtime_load_trace["items"], tick_index=self.tick_index)
            state_snapshot = self.state_pool.snapshot()
            feeling_trace = self._merge_feelings_with_runtime_load(feeling_trace, runtime_load_trace)
        next_budget_trace = self.runtime_budget_controller.observe_runtime_load(runtime_load_trace)
        runtime_budget_trace["next_budget"] = next_budget_trace
        runtime_load_trace["budget_controller"] = {
            "schema_id": "runtime_load_budget_controller_link/v1",
            "applies_to": "next_tick",
            "next_budget": next_budget_trace,
        }
        mark_stage("runtime_load_feeling")
        tick_end_innate_trace = self._evaluate_innate_phase(
            "tick_end",
            state_items=state_snapshot["items"],
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            attention=attention_trace,
            feelings=feeling_trace,
            prediction_trace=self.state_pool.prediction_trace(),
            residual_summary=self.state_pool.residual_summary(limit=8),
            time_trace=time_trace,
            rhythm_trace=rhythm_trace,
            expectation_pressure=expectation_pressure_trace,
            emotion_state=emotion_update_trace.get("emotion_state", {}),
            action_trace=action_trace,
            action_feedback_trace=action_feedback_trace,
            action_consequence_trace=action_consequence_trace,
            runtime_load_trace=runtime_load_trace,
        )
        innate_traces["tick_end"] = tick_end_innate_trace
        if tick_end_innate_trace["items"]:
            self.state_pool.apply_external_items(tick_end_innate_trace["items"], tick_index=self.tick_index)
            state_snapshot = self.state_pool.snapshot()
        self._remember_innate_attention_biases(innate_traces)
        mark_stage("innate_tick_end")
        innate_learning_router_trace = self.innate_learning_router.route(
            tick_index=self.tick_index,
            innate_traces=innate_traces,
            expectation_anchor_trace=expectation_anchor_trace,
            action_feedback_trace=action_feedback_trace,
        )
        mark_stage("innate_learning_router")

        action_causal_window = self._build_action_causal_window(
            selected_actions=list(action_trace.get("selected_actions", []) or []),
            state_snapshot_before_action=state_snapshot_before_action,
            feedback_context=dict(action_trace.get("feedback_context", {}) or {}),
            control_items=control_items,
            action_items=list(action_trace.get("action_items", []) or []),
            text_output_trace=text_output_trace,
            state_snapshot_after_output=state_snapshot,
        )
        action_trace["causal_window"] = action_causal_window
        self._pending_action_feedback = {
            "selected_actions": list(action_trace.get("selected_actions", []) or []),
            "feedback_context": dict(action_trace.get("feedback_context", {}) or {}),
            "causal_window": dict(action_causal_window),
        }
        self._write_memory_snapshots(
            state_snapshot,
            input_packet["normalized_text"],
            attention_trace["selected_labels"],
            asset_refs=multimodal_trace.get("asset_refs", []),
            state_snapshot_for_memory=state_snapshot_for_memory,
        )
        elapsed_before_index_ms = (perf_counter() - stage_started_at_ref()) * 1000.0 + sum(float(stage.get("ms", 0.0) or 0.0) for stage in performance_stages)
        remaining_before_target_ms = target_tick_ms - elapsed_before_index_ms
        index_budget = self.runtime_budget_controller.index_budget(
            base_jobs=self.config.memory.index_jobs_per_tick,
            base_min_remaining_ms=self.config.memory.index_maintenance_min_remaining_ms,
            base_max_ms=self.config.memory.index_maintenance_max_ms,
        )
        runtime_budget_trace["index_budget"] = index_budget
        if int(index_budget.get("jobs_per_tick", 0) or 0) > 0 and remaining_before_target_ms >= float(index_budget["min_remaining_ms"]):
            index_maintenance_trace = self.memory.process_pending_index_jobs(
                int(index_budget["jobs_per_tick"]),
                max_ms=float(index_budget["max_ms"]),
            )
            index_maintenance_trace["policy"] = "time_budgeted_runtime_indexing"
        else:
            index_maintenance_trace = self.memory.process_pending_index_jobs(0)
            index_maintenance_trace["policy"] = "runtime_budget_deferred_indexing" if int(index_budget.get("jobs_per_tick", 0) or 0) <= 0 else "deferred_runtime_indexing"
        index_maintenance_trace["remaining_before_target_ms"] = round(remaining_before_target_ms, 4)
        index_maintenance_trace["runtime_budget"] = index_budget
        runtime_budget_trace["trace_budget"] = self.runtime_budget_controller.trace_budget(
            base_item_preview_limit=self.config.observability.trace_item_preview_limit,
            base_r_state_preview_limit=self.config.observability.trace_r_state_item_preview_limit,
            base_matched_token_limit=self.config.observability.trace_matched_token_preview_limit,
        )
        mark_stage("snapshot_and_memory_write")
        if trace_mode == "debug":
            trace_fast_bn = self.memory.strip_runtime_snapshots(fast_bn)
            trace_slow_bn = self.memory.strip_runtime_snapshots(slow_bn)
            explainability = self._build_explainability(
                state_snapshot=state_snapshot,
                fast_bn=trace_fast_bn,
                fast_cn=fast_cn,
                slow_bn=trace_slow_bn,
                slow_cn=slow_cn,
                attention_trace=attention_trace,
                feeling_trace=feeling_trace,
                runtime_load_trace=runtime_load_trace,
                expectation_pressure_trace=expectation_pressure_trace,
                time_trace=time_trace,
                rhythm_trace=rhythm_trace,
                action_trace=action_trace,
                action_feedback_trace=action_feedback_trace,
                action_consequence_trace=action_consequence_trace,
                text_output_trace=text_output_trace,
                emotion_update_trace=emotion_update_trace,
                emotion_modulation=emotion_modulation,
                prior_emotion_modulation=prior_emotion_modulation,
                focus_continuation_trace=focus_continuation_trace,
                innate_traces=innate_traces,
            )
            thought_view = self._build_thought_view(
                fast_bn=trace_fast_bn,
                fast_cn=fast_cn,
                slow_bn=trace_slow_bn,
                slow_cn=slow_cn,
                attention_trace=attention_trace,
                feeling_trace=feeling_trace,
                runtime_load_trace=runtime_load_trace,
                focus_continuation_trace=focus_continuation_trace,
                expectation_pressure_trace=expectation_pressure_trace,
                text_output_trace=text_output_trace,
            )
            trace = {
                "tick_index": self.tick_index,
                "input": input_packet,
                "competition": competition,
                "multimodal": multimodal_trace,
                "education_intervention": education_intervention_trace,
                "dialogue_turn": dialogue_turn_trace,
                "short_term_echo": short_term_echo_trace,
                "short_term_memory": short_term_memory_trace,
                "short_term_slot": short_term_slot_trace,
                "state_pool": {
                    "r_state": r_state_fast,
                    "query_view": fast_query,
                    "attention_view": attention_candidates,
                    "snapshot": state_snapshot,
                    "energy_flow": dict(state_snapshot.get("energy_flow", {}) or {}),
                },
                "fast_system": {
                    "bn": trace_fast_bn,
                    "cn": fast_cn,
                },
                "attention": attention_trace,
                "slow_system": {
                    "query": slow_query,
                    "bn_prime": trace_slow_bn,
                    "cn_prime": slow_cn,
                    "focus_continuation": focus_continuation_trace,
                    "successor_bias": successor_bias_trace,
                    "successor_bias_update": successor_bias_update_trace,
                },
                "cognitive_feelings": feeling_trace,
                "task_feeling": task_feeling_trace,
                "runtime_load_feeling": runtime_load_trace,
                "runtime_budget_controller": runtime_budget_trace,
                "time_feeling": time_trace,
                "rhythm": rhythm_trace,
                "expectation_pressure": expectation_pressure_trace,
                "emotion": {
                    "update": emotion_update_trace,
                    "modulation": emotion_modulation,
                },
                "innate_rules": self._compact_innate_traces(innate_traces),
                "action": action_trace,
                "action_feedback": action_feedback_trace,
                "text_output": text_output_trace,
                "thought_view": thought_view,
                "explainability": explainability,
                "learning": {
                    "online_embedding": self.memory.online_embedding_summary(),
                    "innate_event_router": innate_learning_router_trace,
                    "index_maintenance": index_maintenance_trace,
                    "runtime_budget_controller": runtime_budget_trace,
                },
                "performance": {
                    "target_tick_ms": target_tick_ms,
                    "stages_ms": performance_stages,
                    "total_ms": round(sum(float(stage.get("ms", 0.0) or 0.0) for stage in performance_stages), 4),
                },
            }
            shaped = self._shape_trace(trace, trace_mode=trace_mode)
        else:
            trace_fast_bn = self.memory.strip_runtime_snapshots(fast_bn)
            trace_slow_bn = self.memory.strip_runtime_snapshots(slow_bn)
            explainability = self._build_runtime_explainability_refs(
                state_snapshot=state_snapshot,
                fast_bn=trace_fast_bn,
                slow_bn=trace_slow_bn,
                attention_trace=attention_trace,
                feeling_trace=feeling_trace,
                runtime_load_trace=runtime_load_trace,
                expectation_pressure_trace=expectation_pressure_trace,
                action_trace=action_trace,
                action_consequence_trace=action_consequence_trace,
                emotion_update_trace=emotion_update_trace,
                emotion_modulation=emotion_modulation,
                prior_emotion_modulation=prior_emotion_modulation,
                text_output_trace=text_output_trace,
                focus_continuation_trace=focus_continuation_trace,
                innate_traces=innate_traces,
            )
            thought_view = self._build_runtime_thought_refs(
                fast_bn=trace_fast_bn,
                fast_cn=fast_cn,
                slow_bn=trace_slow_bn,
                slow_cn=slow_cn,
                attention_trace=attention_trace,
                feeling_trace=feeling_trace,
                runtime_load_trace=runtime_load_trace,
                focus_continuation_trace=focus_continuation_trace,
                expectation_pressure_trace=expectation_pressure_trace,
                text_output_trace=text_output_trace,
            )
            shaped = self._build_summary_trace(
                input_packet=input_packet,
                competition=competition,
                multimodal_trace=multimodal_trace,
                education_intervention_trace=education_intervention_trace,
                dialogue_turn_trace=dialogue_turn_trace,
                short_term_echo_trace=short_term_echo_trace,
                short_term_memory_trace=short_term_memory_trace,
                short_term_slot_trace=dict(getattr(self, "_last_short_term_slot_trace", None) or {}),
                r_state_fast=r_state_fast,
                fast_query=fast_query,
                attention_candidates=attention_candidates,
                state_snapshot=state_snapshot,
                fast_bn=trace_fast_bn,
                fast_cn=fast_cn,
                attention_trace=attention_trace,
                slow_query=slow_query,
                slow_bn=trace_slow_bn,
                slow_cn=slow_cn,
                focus_continuation_trace=focus_continuation_trace,
                successor_bias_trace=successor_bias_trace,
                successor_bias_update_trace=successor_bias_update_trace,
                feeling_trace=feeling_trace,
                task_feeling_trace=task_feeling_trace,
                runtime_load_trace=runtime_load_trace,
                runtime_budget_trace=runtime_budget_trace,
                time_trace=time_trace,
                rhythm_trace=rhythm_trace,
                expectation_pressure_trace=expectation_pressure_trace,
                emotion_update_trace=emotion_update_trace,
                emotion_modulation=emotion_modulation,
                action_trace=action_trace,
                action_feedback_trace=action_feedback_trace,
                text_output_trace=text_output_trace,
                innate_traces=innate_traces,
                thought_view=thought_view,
                explainability=explainability,
                index_maintenance_trace=index_maintenance_trace,
                innate_learning_router_trace=innate_learning_router_trace,
                performance_stages=performance_stages,
            )
        final_start = perf_counter()
        shaping_ms = round((final_start - stage_started_at_ref()) * 1000.0, 4)
        shaped.setdefault("performance", {})
        shaped["performance"]["trace_shape_ms"] = shaping_ms
        shaped["performance"]["total_ms"] = round(float(shaped["performance"].get("total_ms", 0.0) or 0.0) + shaping_ms, 4)
        tuner_trace = self.adaptive_tuner.observe_tick(shaped)
        shaped["tuner"] = tuner_trace
        return shaped

    def _normalize_trace_mode(self, trace_mode: str | None) -> str:
        mode = str(trace_mode or self.config.observability.default_trace_mode or "summary").strip().lower()
        if mode in {"debug", "full", "raw"}:
            return "debug"
        return "summary"

    def _apply_short_term_echo(self, external_items: list[dict]) -> dict:
        """
        Bring recent sensory/thought residues into this tick without faking input.

        Echo items are intentionally applied after true external input and before
        fast recall. They can influence Bn/Cn like a human afterimage or recent
        thought, but their `source_type` and metadata say they are not a new
        external event. The echo buffer only observes the current external items
        after it has built this tick's echoes, so a fresh input does not echo
        itself until a later tick.
        """

        if not bool(getattr(self.config.short_term, "echo_enabled", True)):
            return {
                "schema_id": "short_term_echo_trace/v1",
                "tick_index": int(self.tick_index),
                "applied": False,
                "echo_count": 0,
                "source_counts": {},
                "items": [],
                "items_preview": [],
                "policy": "disabled",
            }
        trace = self.short_term_echo.build_echo_items(tick_index=self.tick_index)
        echo_items = list(trace.get("items", []) or [])
        if echo_items:
            self.state_pool.apply_external_items(echo_items, tick_index=self.tick_index)
        self.short_term_echo.observe_sensory_items(external_items, tick_index=self.tick_index)
        return trace

    def _refresh_dialogue_turn_state(self, *, input_packet: dict, external_items: list[dict]) -> dict:
        normalized = str((input_packet or {}).get("normalized_text", "") or "").strip()
        if normalized:
            self._dialogue_turn_serial += 1
            text_labels = [
                str(item.get("sa_label", "") or "")
                for item in list(external_items or [])[:12]
                if isinstance(item, dict) and str(item.get("sa_label", "") or "").startswith("text::")
            ]
            closure_need = 0.46 + min(0.24, len(normalized) / 120.0)
            if any(mark in normalized for mark in ("?", "？")):
                closure_need += 0.08
            if len(normalized) <= 8:
                closure_need += 0.04
            self._pending_dialogue_turn = {
                "schema_id": "runtime_pending_dialogue_turn/v1",
                "turn_serial": int(self._dialogue_turn_serial),
                "opened_tick": int(self.tick_index),
                "last_refresh_tick": int(self.tick_index),
                "source": "external_text_turn",
                "text_preview": normalized[:80],
                "text_length": len(normalized),
                "external_text_labels": text_labels,
                "closure_need": _round4(min(0.82, closure_need)),
                "policy": "short_lived_process_state_no_reply_text_no_keyword_route",
            }
        pending = dict(self._pending_dialogue_turn or {})
        if not pending:
            return {
                "schema_id": "dialogue_turn_runtime_trace/v1",
                "active": False,
                "opened": False,
                "items": [],
                "policy": "no_current_external_turn",
            }
        pending["last_refresh_tick"] = int(self.tick_index)
        try:
            opened_tick = int(pending.get("opened_tick", self.tick_index))
        except (TypeError, ValueError):
            opened_tick = int(self.tick_index)
        age = max(0, int(self.tick_index) - opened_tick)
        base_need = _round4(float(pending.get("closure_need", 0.0) or 0.0) * max(0.42, 1.0 - age * 0.025))
        meta = {
            **pending,
            "age_ticks": int(age),
            "reply_closure_need": base_need,
            "target_labels": ["text_action::draft_state", "action::text_insert", "action::text_commit"],
            "target_text": "",
            "strictness": 0.12,
        }
        items = [
            {
                "sa_label": "task::reply_to_current_user_turn",
                "display_text": "需要回应当前用户输入",
                "family": "task",
                "source_type": "dialogue_turn_state",
                "real_energy": base_need,
                "virtual_energy": 0.10,
                "cognitive_pressure": _round4(base_need * 0.36),
                "anchor_meta": dict(meta),
            },
            {
                "sa_label": "intention::dialogue_turn_closure",
                "display_text": "本轮对话需要形成闭合",
                "family": "intention",
                "source_type": "dialogue_turn_state",
                "real_energy": _round4(max(0.24, base_need - 0.08)),
                "virtual_energy": 0.08,
                "cognitive_pressure": _round4(base_need * 0.24),
                "anchor_meta": {**meta, "strictness": 0.10},
            },
        ]
        return {
            "schema_id": "dialogue_turn_runtime_trace/v1",
            "active": True,
            "opened": bool(normalized),
            "turn_serial": int(pending.get("turn_serial", 0) or 0),
            "age_ticks": int(age),
            "closure_need": base_need,
            "text_preview": str(pending.get("text_preview", "") or ""),
            "items": items,
            "policy": "pending_dialogue_turn_is_process_state_not_reply_content",
        }

    def _close_dialogue_turn_if_committed(self, text_output_trace: dict) -> dict:
        events = list((text_output_trace or {}).get("recent_events", []) or [])
        commit_events = [
            dict(event)
            for event in events
            if isinstance(event, dict)
            and str(event.get("event_type", "") or "") == "commit"
            and str(event.get("token", "") or "")
        ]
        if not commit_events:
            return {
                "schema_id": "dialogue_turn_close_trace/v1",
                "closed": False,
                "active_before": bool(self._pending_dialogue_turn),
                "reason": "no_nonempty_text_commit",
            }
        prior = dict(self._pending_dialogue_turn or {})
        self._pending_dialogue_turn = None
        return {
            "schema_id": "dialogue_turn_close_trace/v1",
            "closed": True,
            "turn_serial": int(prior.get("turn_serial", 0) or 0),
            "closed_tick": int(self.tick_index),
            "commit_preview": str(commit_events[-1].get("token", "") or "")[:80],
            "policy": "commit_event_closes_pending_dialogue_turn",
        }

    def _observe_short_term_thought_echo(self, ordered_focus_items: list[dict]) -> None:
        if not bool(getattr(self.config.short_term, "echo_enabled", True)):
            return
        self.short_term_echo.observe_thought_items(ordered_focus_items, tick_index=self.tick_index)

    def _observe_short_term_memory(
        self,
        items: list[dict],
        *,
        source_kind: str,
        modality: str | None = None,
        role: str | None = None,
    ) -> dict:
        """
        Store a compact recent-experience event without replaying it.

        This is the P1-J-16 working-memory window. It records many recent
        multimodal facts, while active recall remains action-triggered and
        partial, preserving AP's freedom to ignore, resume, or reinterpret.
        """

        if not bool(getattr(self.config.short_term, "memory_window_enabled", True)):
            return {"schema_id": "short_term_memory_observe_trace/v1", "stored": False, "reason": "disabled"}
        return self.short_term_memory.observe(
            items,
            tick_index=self.tick_index,
            source_kind=source_kind,
            modality=modality,
            role=role,
        )

    def _short_term_memory_similarity(self, query_tokens: list[str], candidate_tokens: list[str]) -> dict:
        return self.memory.learned_similarity(
            query_tokens,
            candidate_tokens,
            limit=self.config.online_embedding.scoring_token_limit,
        )

    def _short_term_memory_trace(self, *, last_recall: dict | None = None) -> dict:
        if not bool(getattr(self.config.short_term, "memory_window_enabled", True)):
            return {
                "schema_id": "short_term_memory_window_trace/v1",
                "tick_index": int(self.tick_index),
                "window_size": 0,
                "active_event_count": 0,
                "last_recall": dict(last_recall or {"available": False}),
                "policy": "disabled",
            }
        return self.short_term_memory.trace(tick_index=self.tick_index, last_recall=last_recall)

    def _short_term_memory_recall_for_actions(
        self,
        *,
        selected_actions: list[dict],
        attention_trace: dict,
        state_snapshot_items: list[dict],
        focus_continuation_trace: dict,
        reason: str,
        update_fatigue: bool,
    ) -> dict:
        if not bool(getattr(self.config.short_term, "memory_window_enabled", True)):
            return {"available": False, "policy": "disabled"}
        action_ids = {str((row or {}).get("action_id", "") or "") for row in list(selected_actions or []) if isinstance(row, dict)}
        if not ({"action::recall_recent_context", "action::replay_recent_context"} & action_ids):
            return {"available": False, "policy": "no_recent_context_action_selected"}
        no_param_recall = False
        for row in list(selected_actions or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") not in {"action::recall_recent_context", "action::replay_recent_context"}:
                continue
            params = dict(row.get("params", {}) or {})
            if str(params.get("recall_mode", "") or "") in {"no_param_recent_context", "unfinished_soft_recovery"}:
                no_param_recall = True
                break
        cues: list[dict] = []
        readback = dict((focus_continuation_trace or {}).get("recent_thought_readback", {}) or {})
        if not no_param_recall:
            for item in list((attention_trace or {}).get("selected_items", []) or [])[: self.config.attention.focus_limit]:
                if isinstance(item, dict):
                    cues.append(dict(item))
            for label in list(readback.get("labels", []) or [])[:8]:
                clean = str(label or "")
                if clean:
                    cues.append({"sa_label": clean, "family": "recent_thought"})
        state_by_label = {
            str(item.get("sa_label", "") or ""): dict(item)
            for item in list(state_snapshot_items or [])
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        }
        enriched = []
        seen = set()
        for cue in cues:
            label = str((cue or {}).get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            enriched.append(state_by_label.get(label, dict(cue)))
        return self.short_term_memory.recall(
            tick_index=self.tick_index,
            cues=enriched,
            limit=self.config.short_term.memory_window_recall_limit,
            horizon_ticks=self.config.short_term.memory_window_max_age_ticks,
            reason=f"{reason}:no_param" if no_param_recall else reason,
            similarity_fn=self._short_term_memory_similarity,
            update_fatigue=update_fatigue,
        )

    def _mark_unfinished_thought_if_needed(
        self,
        *,
        focus_continuation_trace: dict,
        expected_text: dict,
        action_trace: dict,
        short_term_memory_recall: dict,
    ) -> dict:
        if not bool(getattr(self.config.short_term, "memory_window_enabled", True)):
            return {"schema_id": "short_term_unfinished_mark_trace/v1", "stored": False, "reason": "short_term_memory_disabled"}
        selected_ids = {str((row or {}).get("action_id", "") or "") for row in list((action_trace or {}).get("selected_actions", []) or []) if isinstance(row, dict)}
        if {"action::text_insert", "action::continue_focus", "action::recall_recent_context"} & selected_ids:
            return {"schema_id": "short_term_unfinished_mark_trace/v1", "stored": False, "reason": "continuation_or_recall_already_selected"}
        current_labels = [str(label or "") for label in list((focus_continuation_trace or {}).get("current_labels", []) or []) if str(label or "")]
        readback = dict((focus_continuation_trace or {}).get("recent_thought_readback", {}) or {})
        branch_end = max(0.0, float(readback.get("branch_end_score", 0.0) or 0.0))
        drift = max(0.0, float(readback.get("drift_score", 0.0) or 0.0))
        clarity = max(
            0.0,
            float((expected_text or {}).get("strength", 0.0) or 0.0) * 0.34
            + float((expected_text or {}).get("top_share", 0.0) or 0.0) * 0.30
            + float((expected_text or {}).get("dominance_gap", 0.0) or 0.0) * 0.42
            + (0.16 if bool((expected_text or {}).get("decisive", False)) else 0.0),
        )
        recall_available = bool((short_term_memory_recall or {}).get("available", False))
        mark_strength = min(1.0, clarity * 0.72 + max(branch_end, drift) * 0.18 + (0.08 if recall_available else 0.0))
        if mark_strength < float(getattr(self.config.task_feeling, "unfinished_mark_min_strength", 0.18) or 0.18):
            return {
                "schema_id": "short_term_unfinished_mark_trace/v1",
                "stored": False,
                "reason": "not_clear_or_not_interrupted",
                "clarity": round(clarity, 4),
                "branch_end": round(branch_end, 4),
                "drift": round(drift, 4),
            }
        successor_labels = []
        token = str((expected_text or {}).get("token", "") or "")
        if token:
            successor_labels.append(f"text::{token}")
        for alt in list((expected_text or {}).get("alternatives", []) or [])[:4]:
            alt_token = str((alt or {}).get("token", "") or "")
            if alt_token:
                successor_labels.append(f"text::{alt_token}")
        return self.short_term_memory.mark_unfinished(
            tick_index=self.tick_index,
            labels=current_labels or list(readback.get("labels", []) or []),
            successor_labels=successor_labels,
            strength=mark_strength,
            reason="clear_successor_lost_action_competition_or_interruption",
        )

    def _derive_task_feeling(
        self,
        *,
        input_packet: dict,
        fast_cn: list[dict],
        slow_cn: list[dict],
        focus_continuation_trace: dict,
        short_term_memory_trace: dict,
        cognitive_feelings: dict,
        residual_summary: dict,
        action_trace: dict,
    ) -> dict:
        if not bool(getattr(self.config.task_feeling, "enabled", True)):
            return {"schema_id": "task_feeling_trace/v1", "tick_index": int(self.tick_index), "channels": {}, "items": [], "policy": "disabled"}
        expected_text = self.action_planner.expected_text_context(
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            draft_context=self.text_actuator.draft_state(current_tick=self.tick_index),
        )
        return self.task_feeling.derive(
            tick_index=self.tick_index,
            input_packet=input_packet,
            expected_text=expected_text,
            focus_continuation_trace=focus_continuation_trace,
            short_term_memory_trace=short_term_memory_trace,
            cognitive_feelings=cognitive_feelings,
            residual_summary=residual_summary,
            action_trace=action_trace,
        )

    def _trace_item_preview_limit(self) -> int:
        return self.runtime_budget_controller.trace_limit(
            base_limit=self.config.observability.trace_item_preview_limit,
            minimum=4,
        )

    def _trace_r_state_item_preview_limit(self) -> int:
        return self.runtime_budget_controller.trace_limit(
            base_limit=self.config.observability.trace_r_state_item_preview_limit,
            minimum=1,
        )

    def _trace_matched_token_preview_limit(self) -> int:
        return self.runtime_budget_controller.trace_limit(
            base_limit=self.config.observability.trace_matched_token_preview_limit,
            minimum=1,
        )

    def _r_state_to_query_items(self, r_state: dict) -> list[dict]:
        """
        Convert fixed-budget R_state heads into query_items for MemoryStore.recall().

        Policy:
        - Merge heads (dedup by sa_label).
        - Keep the highest query_weight seen across heads.
        - Preserve dual-energy fields for state_match.
        """

        merged: dict[str, dict] = {}
        for head in r_state.get("heads", []) or []:
            for row in head.get("items", []) or []:
                label = str((row or {}).get("sa_label", "") or "")
                if not label:
                    continue
                existing = merged.get(label)
                if existing is None:
                    merged[label] = dict(row)
                    continue
                existing["query_weight"] = max(float(existing.get("query_weight", 0.0) or 0.0), float(row.get("query_weight", 0.0) or 0.0))
                existing["real_energy"] = max(float(existing.get("real_energy", 0.0) or 0.0), float(row.get("real_energy", 0.0) or 0.0))
                existing["virtual_energy"] = max(float(existing.get("virtual_energy", 0.0) or 0.0), float(row.get("virtual_energy", 0.0) or 0.0))
                current_sources = {
                    str(item or "")
                    for item in list(existing.get("current_source_types", []) or []) + list((row or {}).get("current_source_types", []) or [])
                    if str(item or "")
                }
                if current_sources:
                    existing["current_source_types"] = sorted(current_sources)
                    existing["current_tick_item"] = True
        rows = list(merged.values())
        rows = self._apply_query_currentness_salience(rows)
        rows.sort(
            key=lambda item: (
                int(item.get("last_seen_tick", item.get("tick_index", 0)) or 0),
                int(item.get("position", 0) or 0),
                str(item.get("sa_label", "") or ""),
            )
        )
        return rows

    def _apply_query_currentness_salience(self, rows: list[dict]) -> list[dict]:
        """Soft currentness shaping of recall query items.

        Diagnosis (2026-06-13 skill-recall probe): the recall query was dominated
        by accumulated internal residue. The PRIMARY fix lives in the recall
        scoring layer (MemoryStore: specificity-weighted energy/label overlap), so
        rare high-specificity skill anchors win over abundant low-specificity
        generic residue -- per AP philosophy (prediction specificity, not shared-
        label count, drives recall).

        This engine-side hook only applies the legitimate short-term currentness
        principle: the current turn weighs more, history decays but never
        vanishes. It does NOT try to filter/exclude residue as a retrieval key
        (that would overstep into the scoring layer's job and could erase AP's
        cross-domain intuition path). It is soft, reversible, and keys off SA
        source/family + age, never off any natural-language token.
        """
        if not rows:
            return rows
        current_tick = int(self.tick_index or 0)
        FRESH_GAIN = 1.35        # current external turn weighs a bit more
        AGE_DECAY = float(self.config.state_pool.real_decay)
        STALE_FLOOR = float(self.config.memory.temporal_floor)
        new_external_turn = False
        active_external_turn = False
        pending = dict(getattr(self, "_pending_dialogue_turn", {}) or {})
        active_external_turn = bool(pending)
        try:
            new_external_turn = bool(int(pending.get("opened_tick", -1) or -1) == current_tick)
        except (TypeError, ValueError):
            new_external_turn = False
        current_turn_sources = {
            "external_text",
            "external_text_readback",
            "external_teacher",
            "dialogue_turn_state",
            "vision_bridge",
            "audio_bridge",
            "vision_numeric",
            "audio_numeric",
        }
        short_residue_sources = {
            "predicted",
            "text_action",
            "internal_draft_visible",
            "internal_draft_read",
            "short_term_slot",
            "short_term_echo",
            "thought_echo",
            "action_control",
            "action_feedback",
            "time_feeling",
            "rhythm_feeling",
        }
        short_residue_families = {
            "text_action",
            "text_revision_opportunity",
            "short_term_slot",
            "action_control",
            "action_feedback",
            "time_feeling",
            "rhythm_feeling",
            "expectation_pressure",
        }
        active_turn_residue_sources = {
            "predicted",
            "short_term_slot",
            "short_term_echo",
            "thought_echo",
            "action_control",
            "action_feedback",
            "time_feeling",
            "rhythm_feeling",
            "expectation_pressure",
        }
        active_turn_residue_families = {
            "short_term_slot",
            "action_control",
            "action_feedback",
            "time_feeling",
            "rhythm_feeling",
            "expectation_pressure",
        }
        for row in rows:
            source_type = str(row.get("source_type", "") or "")
            family = str(row.get("family", "") or "")
            current_sources = {
                str(item or "")
                for item in list(row.get("current_source_types", []) or [])
                if str(item or "")
            }
            last_seen = int(row.get("last_seen_tick", row.get("tick_index", current_tick)) or current_tick)
            age = max(0, current_tick - last_seen)
            base_weight = float(row.get("query_weight", row.get("real_energy", 0.0)) or 0.0)

            is_current_turn_source = bool(current_sources & current_turn_sources)
            is_fresh = age <= 0
            is_short_residue = bool(
                source_type in short_residue_sources
                or family in short_residue_families
                or (current_sources and not (current_sources & current_turn_sources))
            )
            is_active_turn_internal_residue = bool(
                source_type in active_turn_residue_sources
                or family in active_turn_residue_families
                or (current_sources and not (current_sources & current_turn_sources))
            )

            if is_fresh and is_current_turn_source:
                factor = FRESH_GAIN
            elif (new_external_turn and is_short_residue) or (active_external_turn and is_active_turn_internal_residue):
                # A new user turn should draw attention back to the current
                # high-pressure field. Recently predicted/action-feedback
                # residues stay available, but they no longer masquerade as
                # current external evidence simply because their SA label was
                # touched again by Cn or short-term echo. During the active
                # turn, self-observed draft surface can still drive charwise
                # continuation; internal consequence residue remains background
                # unless it earns attention through the ordinary action loop.
                base_decay = AGE_DECAY ** max(1, age)
                factor = max(STALE_FLOOR, base_decay * (1.0 - float(self.config.memory.temporal_fatigue_strength) * 0.55))
            else:
                factor = max(STALE_FLOOR, AGE_DECAY ** age) if age > 0 else 1.0

            row["query_weight"] = base_weight * factor
            row["query_currentness"] = {
                "age": age,
                "factor": round(factor, 4),
                "fresh_external": bool(is_fresh and is_current_turn_source),
                "current_source_types": sorted(current_sources),
                "new_external_turn_residue_softened": bool(new_external_turn and is_short_residue and not (is_fresh and is_current_turn_source)),
                "active_external_turn_residue_softened": bool(active_external_turn and is_active_turn_internal_residue and not (is_fresh and is_current_turn_source)),
            }
        return rows

    def _evaluate_innate_phase(
        self,
        phase: str,
        *,
        state_items: list[dict] | None = None,
        fast_bn: list[dict] | None = None,
        fast_cn: list[dict] | None = None,
        slow_bn: list[dict] | None = None,
        slow_cn: list[dict] | None = None,
        attention: dict | None = None,
        feelings: dict | None = None,
        prediction_trace: dict | None = None,
        residual_summary: dict | None = None,
        time_trace: dict | None = None,
        rhythm_trace: dict | None = None,
        expectation_pressure: dict | None = None,
        emotion_state: dict | None = None,
        action_trace: dict | None = None,
        action_feedback_trace: dict | None = None,
        action_consequence_trace: dict | None = None,
        runtime_load_trace: dict | None = None,
        draft_context: dict | None = None,
    ) -> dict:
        context = {
            "tick_index": self.tick_index,
            "state_items": list(state_items or []),
            "fast_bn": list(fast_bn or []),
            "fast_cn": list(fast_cn or []),
            "slow_bn": list(slow_bn or []),
            "slow_cn": list(slow_cn or []),
            "attention": dict(attention or {}),
            "feelings": dict(feelings or {}),
            "prediction_trace": dict(prediction_trace or self.state_pool.prediction_trace()),
            "residual_summary": dict(residual_summary or self.state_pool.residual_summary(limit=8)),
            "time_trace": dict(time_trace or {}),
            "rhythm_trace": dict(rhythm_trace or {}),
            "expectation_pressure": dict(expectation_pressure or {}),
            "emotion_state": dict(emotion_state or {}),
            "action_trace": dict(action_trace or {}),
            "action_feedback_trace": dict(action_feedback_trace or {}),
            "action_consequence_trace": dict(action_consequence_trace or {}),
            "runtime_load_trace": dict(runtime_load_trace or {}),
            "draft_context": dict(draft_context or {}),
            "ui_trace": dict((action_trace or {}).get("ui_trace", {}) or {}),
            "pointer_trace": dict((action_trace or {}).get("pointer_trace", {}) or {}),
        }
        return self.innate_engine.evaluate(phase=phase, context=context, tick_index=self.tick_index)

    def _consume_pending_innate_attention_biases(self) -> list[dict]:
        biases = [dict(row) for row in list(self._pending_innate_attention_biases or []) if isinstance(row, dict)]
        self._pending_innate_attention_biases = []
        return biases[:12]

    def _remember_innate_attention_biases(self, innate_traces: dict | None) -> None:
        rows: list[dict] = []
        for phase, trace in dict(innate_traces or {}).items():
            for bias in list((trace or {}).get("attention_biases", []) or []):
                if not isinstance(bias, dict):
                    continue
                row = dict(bias)
                row.setdefault("source_phase", str(phase))
                row["created_tick_index"] = int(self.tick_index)
                rows.append(row)
        self._pending_innate_attention_biases = rows[-12:]

    def _consume_pending_action_attention_controls(self) -> list[dict]:
        controls, remaining = self._consume_ttl_rows(self._pending_action_attention_controls)
        self._pending_action_attention_controls = remaining
        return controls[:12]

    def _enrich_action_attention_controls_with_learned_bands(self, controls: list[dict], *, attention_candidates: list[dict]) -> list[dict]:
        if not controls:
            return []
        enriched: list[dict] = []
        band_cache: dict[tuple, dict] = {}
        token_cache: dict[str, list[str]] = {}
        candidate_token_cache: dict[str, list[str]] = {}
        vector_cache: dict[tuple, list[float]] = {}
        association_cache: dict[tuple, dict] = {}
        score_limit = max(8, int(getattr(self.config.online_embedding, "scoring_token_limit", 32) or 32))
        candidate_limit = max(
            6,
            min(
                96,
                int(getattr(self.config.memory, "candidate_limit", 40) or 40),
                int(getattr(self.config.memory, "scoring_candidate_limit", 24) or 24),
            ),
        )
        for control in list(controls or []):
            if not isinstance(control, dict):
                continue
            row = dict(control)
            if not bool(row.get("learned_band_enabled", True)):
                enriched.append(row)
                continue
            control_kind = str(row.get("control_kind", "") or "")
            if control_kind not in {"focus_anchor", "continue_focus", "inspect_residual", "release_focus", "diverge_attention"}:
                enriched.append(row)
                continue
            anchor_tokens = self._attention_control_anchor_tokens(row)
            if not anchor_tokens:
                enriched.append(row)
                continue
            band_mode = str(row.get("band_mode", "") or "")
            if not band_mode:
                band_mode = "release" if control_kind == "release_focus" else ("diverge" if control_kind == "diverge_attention" else ("hold" if control_kind == "continue_focus" else "narrow"))
            row.setdefault("band_mode", band_mode)
            if band_mode == "diverge":
                row["learned_band_biases"] = []
                row["learned_band_policy"] = {
                    "schema_id": "learned_semantic_attention_band/v1",
                    "enabled": True,
                    "mode": "diverge",
                    "meaning": "diverge_attention_relaxes_learned_band_and_restores_broader_competition",
                }
                enriched.append(row)
                continue
            row.setdefault("band_gain", 0.44 if control_kind in {"focus_anchor", "inspect_residual"} else 0.32)
            row.setdefault("band_suppression_gain", 0.28 if band_mode == "narrow" else 0.18)
            row.setdefault("band_width", 0.20)
            row["band_anchor_tokens"] = anchor_tokens[:8]
            row["learned_band_biases"] = self._learned_attention_band_biases(
                anchor_tokens,
                attention_candidates=attention_candidates,
                band_mode=band_mode,
                band_width=float(row.get("band_width", 0.20) or 0.20),
                cache=band_cache,
                token_cache=token_cache,
                candidate_token_cache=candidate_token_cache,
                vector_cache=vector_cache,
                association_cache=association_cache,
                score_limit=score_limit,
                candidate_limit=candidate_limit,
            )
            row["learned_band_policy"] = {
                "schema_id": "learned_semantic_attention_band/v1",
                "enabled": True,
                "mode": band_mode,
                "anchor_token_count": len(anchor_tokens[:12]),
                "candidate_count": len(attention_candidates or []),
                "bias_count": len(row.get("learned_band_biases", []) or []),
                "source": "MemoryStore.learned_vector_similarity+learned_similarity",
                "boundary": "band_bias_is_soft_attention_modulation_not_keyword_route",
            }
            enriched.append(row)
        return enriched[:12]

    def _attention_control_anchor_tokens(self, control: dict) -> list[str]:
        tokens: list[str] = []
        for key in ("band_anchor_tokens", "anchor_tokens", "boost_labels", "slow_query_labels", "target_labels", "suppress_labels", "release_labels"):
            value = control.get(key)
            if isinstance(value, (list, tuple)):
                tokens.extend(str(item or "") for item in value if str(item or ""))
        for key in ("anchor_label", "target_label", "sa_label"):
            value = control.get(key)
            if isinstance(value, str) and value:
                tokens.append(value)
        return self._unique_text_tokens(tokens, limit=16)

    def _learned_attention_band_biases(
        self,
        anchor_tokens: list[str],
        *,
        attention_candidates: list[dict],
        band_mode: str,
        band_width: float,
        cache: dict[tuple, dict] | None = None,
        token_cache: dict[str, list[str]] | None = None,
        candidate_token_cache: dict[str, list[str]] | None = None,
        vector_cache: dict[tuple, list[float]] | None = None,
        association_cache: dict[tuple, dict] | None = None,
        score_limit: int | None = None,
        candidate_limit: int | None = None,
    ) -> list[dict]:
        anchors = self._unique_text_tokens(anchor_tokens, limit=16)
        if not anchors:
            return []
        scoring_limit = max(8, int(score_limit if score_limit is not None else self.config.online_embedding.scoring_token_limit))
        scan_limit = max(1, int(candidate_limit if candidate_limit is not None else 96))
        band_key = (
            tuple(anchors[:16]),
            str(band_mode or "narrow"),
            round(float(band_width or 0.0), 4),
            scoring_limit,
            scan_limit,
            tuple(str((candidate or {}).get("sa_label", "") or "") for candidate in list(attention_candidates or [])[:scan_limit] if isinstance(candidate, dict)),
        )
        if cache is not None and band_key in cache:
            return [dict(row) for row in list(cache[band_key].get("rows", []) or [])]

        vector_cache = vector_cache if vector_cache is not None else {}
        association_cache = association_cache if association_cache is not None else {}
        candidate_token_cache = candidate_token_cache if candidate_token_cache is not None else {}

        def cached_vector(tokens: list[str]) -> list[float]:
            key = (tuple(tokens[:scoring_limit]), scoring_limit)
            existing = vector_cache.get(key)
            if existing is not None:
                return existing
            vector = list(self.memory.learned_vector(tokens, limit=scoring_limit))
            vector_cache[key] = vector
            return vector

        def vector_similarity(left_tokens: list[str], right_tokens: list[str]) -> dict:
            left = cached_vector(left_tokens)
            right = cached_vector(right_tokens)
            score = sum(float(a or 0.0) * float(b or 0.0) for a, b in zip(left, right))
            left_norm = sum(float(value or 0.0) * float(value or 0.0) for value in left) ** 0.5
            right_norm = sum(float(value or 0.0) * float(value or 0.0) for value in right) ** 0.5
            return {"score": round(score, 4), "query_norm": round(left_norm, 4), "candidate_norm": round(right_norm, 4)}

        def association_similarity(left_tokens: list[str], right_tokens: list[str]) -> dict:
            key = (tuple(left_tokens[:scoring_limit]), tuple(right_tokens[:scoring_limit]), scoring_limit)
            existing = association_cache.get(key)
            if existing is not None:
                return dict(existing)
            value = dict(self.memory.learned_similarity(left_tokens, right_tokens, limit=scoring_limit))
            association_cache[key] = value
            return dict(value)

        def candidate_feedback(candidate: dict) -> dict:
            meta = dict(candidate.get("anchor_meta", {}) or {}) if isinstance(candidate.get("anchor_meta", {}), dict) else {}

            def value_for(*keys: str) -> float:
                values: list[float] = []
                for key in keys:
                    for source in (candidate, meta):
                        try:
                            values.append(float(source.get(key, 0.0) or 0.0))
                        except (TypeError, ValueError):
                            continue
                return max(values) if values else 0.0

            reward = max(0.0, value_for("feedback_reward", "reward_value", "reward"))
            punishment = max(0.0, value_for("feedback_punishment", "punishment_value", "punishment"))
            correctness = max(0.0, value_for("feedback_correctness", "correctness"))
            utility = reward + correctness * 0.35 - punishment * 0.85
            return {
                "reward": round(reward, 4),
                "punishment": round(punishment, 4),
                "correctness": round(correctness, 4),
                "utility": round(utility, 4),
            }

        rows: list[dict] = []
        for candidate in list(attention_candidates or [])[:scan_limit]:
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("sa_label", "") or "")
            if not label:
                continue
            cache_key = label + "\x1f" + str(candidate.get("display_text", "") or "")
            candidate_tokens = candidate_token_cache.get(cache_key)
            if candidate_tokens is None:
                candidate_tokens = self._attention_candidate_tokens(candidate)
                candidate_token_cache[cache_key] = list(candidate_tokens)
            if not candidate_tokens:
                continue
            vector = vector_similarity(anchors, candidate_tokens)
            association = association_similarity(anchors, candidate_tokens)
            vector_score = max(0.0, float(vector.get("score", 0.0) or 0.0))
            association_score = max(0.0, float(association.get("score", 0.0) or 0.0))
            contribution_count = len(list(association.get("contributions", []) or []))
            negative_count = len(list(association.get("negative_contributions", []) or []))
            evidence_count = contribution_count + negative_count
            # Seed vectors can be accidentally close before AP has learned the
            # relation. Association evidence or a strong learned-vector score is
            # required before the band can materially affect attention.
            learned_evidence_gate = 1.0 if evidence_count > 0 else 0.0
            if evidence_count <= 0 and vector_score >= max(0.42, float(band_width) + 0.22):
                learned_evidence_gate = 0.45
            score = max(0.0, association_score * 0.72 + vector_score * 0.28 * learned_evidence_gate)
            if str(band_mode or "") == "release":
                score = max(score, vector_score * 0.18 if evidence_count > 0 else 0.0)
            feedback = candidate_feedback(candidate)
            positive_feedback = max(0.0, float(feedback.get("utility", 0.0) or 0.0))
            negative_feedback = max(0.0, -float(feedback.get("utility", 0.0) or 0.0))
            if score > 0.0 and positive_feedback > 0.0:
                support_ratio = float(evidence_count) / max(1.0, float(evidence_count + 1))
                score *= 1.0 + positive_feedback * (1.0 + support_ratio)
            if score > 0.0 and negative_feedback > 0.0:
                score /= 1.0 + negative_feedback
            if score <= 0.0 and evidence_count <= 0:
                continue
            rows.append(
                {
                    "schema_id": "learned_attention_band_bias/v1",
                    "sa_label": label,
                    "candidate_tokens": candidate_tokens[:10],
                    "anchor_tokens": anchors[:10],
                    "score": round(score, 4),
                    "vector_score": round(vector_score, 4),
                    "association_score": round(association_score, 4),
                    "feedback_reward": float(feedback.get("reward", 0.0) or 0.0),
                    "feedback_punishment": float(feedback.get("punishment", 0.0) or 0.0),
                    "feedback_correctness": float(feedback.get("correctness", 0.0) or 0.0),
                    "feedback_utility": float(feedback.get("utility", 0.0) or 0.0),
                    "evidence_count": int(evidence_count),
                    "positive_contribution_count": int(contribution_count),
                    "negative_contribution_count": int(negative_count),
                    "band_mode": str(band_mode or "narrow"),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        result = rows[:16]
        if cache is not None:
            cache[band_key] = {"rows": [dict(row) for row in result]}
        return result

    def _attention_candidate_tokens(self, item: dict) -> list[str]:
        tokens: list[str] = []
        label = str((item or {}).get("sa_label", "") or "")
        display = str((item or {}).get("display_text", "") or "")
        family = str((item or {}).get("family", "") or "")
        source_type = str((item or {}).get("source_type", "") or "")
        if label:
            tokens.append(label)
        if display:
            tokens.append(display)
            tokens.extend(display.replace("::", " ").replace("_", " ").split())
        for prefix, value in (("family", family), ("source_type", source_type)):
            if value:
                tokens.append(f"{prefix}::{value}")
        meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        for key in ("schema_id", "event_type", "process_anchor_role", "semantic_frame_role", "readout_semantic_role", "feedback_outcome"):
            value = meta.get(key)
            if value is not None and str(value) != "":
                tokens.append(f"{key}::{value}")
        return self._unique_text_tokens(tokens, limit=20)

    def _unique_text_tokens(self, values: list[str], *, limit: int) -> list[str]:
        seen: set[str] = set()
        rows: list[str] = []
        for value in list(values or []):
            clean = str(value or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
            if len(rows) >= max(1, int(limit)):
                break
        return rows

    def _consume_pending_slow_query_hints(self) -> list[dict]:
        hints, remaining = self._consume_ttl_rows(self._pending_slow_query_hints)
        self._pending_slow_query_hints = remaining
        return hints[:12]

    def _consume_pending_focus_family_modulation(self) -> dict:
        row = dict(self._pending_focus_family_modulation or {})
        if not row:
            return {}
        ttl = max(0, int(row.get("ttl", 0) or 0))
        if ttl <= 0:
            self._pending_focus_family_modulation = {}
            return {}
        consumed = dict(row)
        next_row = dict(row)
        next_row["ttl"] = ttl - 1
        self._pending_focus_family_modulation = next_row if next_row["ttl"] > 0 else {}
        return consumed

    def _remember_action_control_effects(self, effect_trace: dict) -> None:
        controls = [dict(row) for row in list((effect_trace or {}).get("attention_controls", []) or []) if isinstance(row, dict)]
        hints = [dict(row) for row in list((effect_trace or {}).get("slow_query_hints", []) or []) if isinstance(row, dict)]
        if controls:
            self._pending_action_attention_controls = (self._pending_action_attention_controls + controls)[-18:]
        if hints:
            self._pending_slow_query_hints = (self._pending_slow_query_hints + hints)[-18:]
        modulation = dict((effect_trace or {}).get("family_budget_modulation", {}) or {})
        if modulation:
            self._pending_focus_family_modulation = modulation

    def _consume_ttl_rows(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        consumed: list[dict] = []
        remaining: list[dict] = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            ttl = max(0, int(row.get("ttl", 0) or 0))
            if ttl <= 0:
                continue
            consumed.append(dict(row))
            next_row = dict(row)
            next_row["ttl"] = ttl - 1
            if next_row["ttl"] > 0:
                remaining.append(next_row)
        return consumed, remaining

    def _shape_trace(self, trace: dict, *, trace_mode: str) -> dict:
        if trace_mode == "debug":
            trace["trace_mode"] = "debug"
            return trace
        shaped = dict(trace)
        shaped["trace_mode"] = "summary"
        shaped["competition"] = self._summarize_competition(trace.get("competition", {}))
        shaped["input"] = self._summarize_input(trace.get("input", {}))
        state_pool = dict(trace.get("state_pool", {}) or {})
        state_pool["r_state"] = self._summarize_r_state(state_pool.get("r_state", {}))
        query_view = list(state_pool.get("query_view", []) or [])
        attention_view = list(state_pool.get("attention_view", []) or [])
        state_pool["query_view_total_count"] = len(query_view)
        state_pool["attention_view_total_count"] = len(attention_view)
        state_pool["query_view"] = self._compact_rows(query_view, limit=self._trace_item_preview_limit())
        state_pool["attention_view"] = self._compact_rows(attention_view, limit=self._trace_item_preview_limit())
        shaped["state_pool"] = state_pool
        attention_trace = dict(trace.get("attention", {}) or {})
        attention_trace["selected_items"] = self._compact_rows(attention_trace.get("selected_items", []), limit=self.config.attention.focus_limit)
        attention_trace["ranked_items"] = self._compact_rows(attention_trace.get("ranked_items", []), limit=self._trace_item_preview_limit())
        shaped["attention"] = attention_trace
        fast_system = dict(trace.get("fast_system", {}) or {})
        fast_system["bn"] = self._compact_bn_rows(fast_system.get("bn", []))
        shaped["fast_system"] = fast_system
        slow_system = dict(trace.get("slow_system", {}) or {})
        slow_system["query"] = self._compact_rows(slow_system.get("query", []), limit=self._trace_item_preview_limit())
        slow_system["bn_prime"] = self._compact_bn_rows(slow_system.get("bn_prime", []))
        if "focus_continuation" in slow_system:
            slow_system["focus_continuation"] = self._compact_focus_continuation_trace(slow_system.get("focus_continuation", {}))
        shaped["slow_system"] = slow_system
        shaped["thought_view"] = self._compact_thought_view(trace.get("thought_view", {}))
        shaped["education_intervention"] = self._compact_education_intervention_trace(trace.get("education_intervention", {}))
        shaped["short_term_echo"] = self._compact_short_term_echo_trace(trace.get("short_term_echo", {}))
        shaped["short_term_memory"] = self._compact_short_term_memory_trace(trace.get("short_term_memory", {}))
        shaped["task_feeling"] = trace.get("task_feeling", {})
        shaped["explainability"] = self._compact_explainability(trace.get("explainability", {}))
        shaped["action"] = self._compact_action_trace(trace.get("action", {}))
        shaped["innate_rules"] = self._compact_innate_traces(trace.get("innate_rules", {}))
        return shaped

    def _build_summary_trace(
        self,
        *,
        input_packet: dict,
        competition: dict,
        multimodal_trace: dict,
        education_intervention_trace: dict,
        dialogue_turn_trace: dict,
        short_term_echo_trace: dict,
        short_term_memory_trace: dict,
        short_term_slot_trace: dict,
        r_state_fast: dict,
        fast_query: list[dict],
        attention_candidates: list[dict],
        state_snapshot: dict,
        fast_bn: list[dict],
        fast_cn: list[dict],
        attention_trace: dict,
        slow_query: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        focus_continuation_trace: dict,
        successor_bias_trace: dict,
        successor_bias_update_trace: dict,
        feeling_trace: dict,
        task_feeling_trace: dict,
        runtime_load_trace: dict,
        runtime_budget_trace: dict,
        time_trace: dict,
        rhythm_trace: dict,
        expectation_pressure_trace: dict,
        emotion_update_trace: dict,
        emotion_modulation: dict,
        action_trace: dict,
        action_feedback_trace: dict,
        text_output_trace: dict,
        innate_traces: dict | None = None,
        thought_view: dict,
        explainability: dict,
        index_maintenance_trace: dict,
        innate_learning_router_trace: dict | None = None,
        performance_stages: list[dict],
    ) -> dict:
        return {
            "trace_mode": "summary",
            "tick_index": self.tick_index,
            "input": self._summarize_input(input_packet),
            "competition": self._summarize_competition(competition),
            "multimodal": multimodal_trace,
            "education_intervention": self._compact_education_intervention_trace(education_intervention_trace),
            "dialogue_turn": dialogue_turn_trace,
            "short_term_echo": self._compact_short_term_echo_trace(short_term_echo_trace),
            "short_term_memory": self._compact_short_term_memory_trace(short_term_memory_trace),
            "short_term_slot": short_term_slot_trace,
            "state_pool": {
                "r_state": self._summarize_r_state(r_state_fast),
                "query_view_total_count": len(fast_query or []),
                "attention_view_total_count": len(attention_candidates or []),
                "query_view": self._compact_rows(fast_query, limit=self._trace_item_preview_limit()),
                "attention_view": self._compact_rows(attention_candidates, limit=self._trace_item_preview_limit()),
                "snapshot": state_snapshot,
                "energy_flow": dict(state_snapshot.get("energy_flow", {}) or {}),
            },
            "fast_system": {
                "bn": self._compact_bn_rows(fast_bn),
                "cn": fast_cn,
            },
            "attention": {
                **{key: value for key, value in (attention_trace or {}).items() if key not in {"selected_items", "ranked_items"}},
                "selected_items": self._compact_rows((attention_trace or {}).get("selected_items", []), limit=self.config.attention.focus_limit),
                "ranked_items": self._compact_rows((attention_trace or {}).get("ranked_items", []), limit=self._trace_item_preview_limit()),
            },
            "slow_system": {
                "query": self._compact_rows(slow_query, limit=self._trace_item_preview_limit()),
                "bn_prime": self._compact_bn_rows(slow_bn),
                "cn_prime": slow_cn,
                "focus_continuation": focus_continuation_trace,
                "successor_bias": successor_bias_trace,
                "successor_bias_update": successor_bias_update_trace,
            },
            "cognitive_feelings": feeling_trace,
            "task_feeling": task_feeling_trace,
            "runtime_load_feeling": runtime_load_trace,
            "runtime_budget_controller": runtime_budget_trace,
            "time_feeling": time_trace,
            "rhythm": rhythm_trace,
            "expectation_pressure": expectation_pressure_trace,
            "emotion": {
                "update": emotion_update_trace,
                "modulation": emotion_modulation,
            },
            "innate_rules": self._compact_innate_traces(innate_traces or {}),
            "action": self._compact_action_trace(action_trace),
            "action_feedback": action_feedback_trace,
            "text_output": text_output_trace,
            "thought_view": self._compact_thought_view(thought_view),
            "explainability": self._compact_explainability(explainability),
            "learning": {
                "online_embedding": self.memory.online_embedding_summary(),
                "innate_event_router": dict(innate_learning_router_trace or {}),
                "index_maintenance": index_maintenance_trace,
                "runtime_budget_controller": runtime_budget_trace,
            },
            "performance": {
                "target_tick_ms": float(getattr(self.config.observability, "target_tick_ms", 100) or 100),
                "stages_ms": performance_stages,
                "total_ms": round(sum(float(stage.get("ms", 0.0) or 0.0) for stage in performance_stages), 4),
            },
        }

    def _compact_short_term_echo_trace(self, trace: dict) -> dict:
        row = dict(trace or {})
        row["items"] = self._compact_short_term_echo_items(list(row.get("items", []) or []), limit=8)
        row["items_preview"] = list(row.get("items_preview", []) or [])[:8]
        return row

    def _compact_education_intervention_trace(self, trace: dict) -> dict:
        row = dict(trace or {})
        row["state_items"] = self._compact_rows(list(row.get("state_items", []) or []), limit=6)
        row["action_biases"] = [
            {
                "action_id": str(bias.get("action_id", "") or ""),
                "drive_delta": bias.get("drive_delta", 0.0),
                "params": dict(bias.get("params", {}) or {}),
                "notes": list(bias.get("notes", []) or [])[:8],
                "teacher_kind": str(bias.get("teacher_kind", "") or ""),
            }
            for bias in list(row.get("action_biases", []) or [])[:8]
            if isinstance(bias, dict)
        ]
        row["interventions"] = [
            {
                "source": str(item.get("source", "") or ""),
                "teacher_kind": str(item.get("teacher_kind", "") or ""),
                "goal": str(item.get("goal", "") or ""),
                "state_item_count": len(list(item.get("state_items", []) or [])),
                "action_bias_count": len(list(item.get("action_biases", []) or [])),
                "has_feedback": bool(item.get("feedback", {})),
            }
            for item in list(row.get("interventions", []) or [])[:6]
            if isinstance(item, dict)
        ]
        return row

    def _compact_short_term_memory_trace(self, trace: dict) -> dict:
        row = dict(trace or {})
        row["recent_events"] = list(row.get("recent_events", []) or [])[:8]
        row["observations"] = list(row.get("observations", []) or [])[:8]
        recall = dict(row.get("last_recall", {}) or {})
        if recall:
            recall["selected_events"] = list(recall.get("selected_events", []) or [])[:4]
            recall["selected_items"] = list(recall.get("selected_items", []) or [])[:8]
            recall["candidate_preview"] = list(recall.get("candidate_preview", []) or [])[:6]
        row["last_recall"] = recall if recall else {"available": False}
        return row

    def _compact_short_term_echo_items(self, rows: list[dict], *, limit: int) -> list[dict]:
        """
        Keep echo rows small while preserving what the observatory needs.

        Summary traces normally drop heavy payloads, but short-term echo is a
        readout of recent afterimage/aftersound residue. The observatory cannot
        draw or synthesize that residue unless the bounded echo trace keeps the
        already-state-pool-derived numeric payload and echo provenance. This is
        still trace-only data; it never feeds back into cognition.
        """

        compact = []
        for source in list(rows or [])[: max(1, int(limit))]:
            if not isinstance(source, dict):
                continue
            item = {
                "sa_label": str(source.get("sa_label", "") or ""),
                "display_text": str(source.get("display_text", "") or ""),
                "family": str(source.get("family", "") or ""),
                "source_type": str(source.get("source_type", "") or ""),
                "real_energy": float(source.get("real_energy", 0.0) or 0.0),
                "virtual_energy": float(source.get("virtual_energy", 0.0) or 0.0),
                "cognitive_pressure": float(source.get("cognitive_pressure", 0.0) or 0.0),
            }
            if isinstance(source.get("anchor_meta"), dict):
                item["anchor_meta"] = dict(source.get("anchor_meta", {}) or {})
            if isinstance(source.get("numeric_features"), dict):
                item["numeric_features"] = {
                    str(channel): list(values if isinstance(values, (list, tuple)) else [values])
                    for channel, values in dict(source.get("numeric_features", {}) or {}).items()
                    if str(channel or "")
                }
            if isinstance(source.get("reconstruction_payload"), dict):
                item["reconstruction_payload"] = dict(source.get("reconstruction_payload", {}) or {})
            compact.append(item)
        return compact

    def process_idle_maintenance(self, *, include_heavy: bool = True, budget: int | None = None, max_ms: float | None = None) -> dict:
        """
        Run non-realtime maintenance under an explicit budget.

        Use this from empty/idle windows, experiments, or future observatory
        buttons. It is deliberately not hidden inside the cognitive tick.
        """

        explicit_budget = budget is not None or max_ms is not None
        if include_heavy and explicit_budget:
            trace = self.memory.process_idle_index_maintenance(
                budget=self.config.memory.idle_heavy_index_jobs if budget is None else budget,
                max_ms=self.config.memory.idle_index_maintenance_max_ms if max_ms is None else max_ms,
            )
            trace["runtime_budget"] = {
                "schema_id": "runtime_index_budget/v1",
                "policy": "explicit_idle_budget_not_modulated",
                "jobs_per_tick": int(self.config.memory.idle_heavy_index_jobs if budget is None else budget),
                "max_ms": float(self.config.memory.idle_index_maintenance_max_ms if max_ms is None else max_ms),
            }
        elif include_heavy:
            budget_trace = self.runtime_budget_controller.index_budget(
                base_jobs=self.config.memory.idle_heavy_index_jobs if budget is None else budget,
                base_min_remaining_ms=0.0,
                base_max_ms=self.config.memory.idle_index_maintenance_max_ms if max_ms is None else max_ms,
            )
            trace = self.memory.process_idle_index_maintenance(
                budget=int(budget_trace["jobs_per_tick"]),
                max_ms=float(budget_trace["max_ms"]),
            )
            trace["runtime_budget"] = budget_trace
        elif explicit_budget:
            trace = self.memory.process_pending_index_jobs(
                self.config.memory.index_jobs_per_tick if budget is None else budget,
                max_ms=self.config.memory.index_maintenance_max_ms if max_ms is None else max_ms,
                include_heavy=False,
            )
            trace["policy"] = "idle_light_index_maintenance"
            trace["runtime_budget"] = {
                "schema_id": "runtime_index_budget/v1",
                "policy": "explicit_idle_budget_not_modulated",
                "jobs_per_tick": int(self.config.memory.index_jobs_per_tick if budget is None else budget),
                "max_ms": float(self.config.memory.index_maintenance_max_ms if max_ms is None else max_ms),
            }
        else:
            budget_trace = self.runtime_budget_controller.index_budget(
                base_jobs=self.config.memory.index_jobs_per_tick if budget is None else budget,
                base_min_remaining_ms=0.0,
                base_max_ms=self.config.memory.index_maintenance_max_ms if max_ms is None else max_ms,
            )
            trace = self.memory.process_pending_index_jobs(
                int(budget_trace["jobs_per_tick"]),
                max_ms=float(budget_trace["max_ms"]),
                include_heavy=False,
            )
            trace["policy"] = "idle_light_index_maintenance"
            trace["runtime_budget"] = budget_trace
        if bool(self.config.observability.disable_gc_during_tick):
            generation = max(0, int(self.config.observability.idle_gc_collect_generation))
            gc_started = perf_counter()
            trace["gc_collected"] = int(gc.collect(generation))
            trace["gc_ms"] = round((perf_counter() - gc_started) * 1000.0, 4)
        return trace

    def _summarize_input(self, input_packet: dict) -> dict:
        packet = dict(input_packet or {})
        units = list(packet.get("units", []) or [])
        packet["units"] = self._compact_rows(units, limit=self._trace_item_preview_limit())
        sa_items = list(packet.get("sa_items", []) or [])
        packet["sa_item_count"] = int(packet.get("sa_item_count", len(sa_items)) or 0)
        if sa_items:
            packet["sa_items"] = self._compact_rows(sa_items, limit=self._trace_item_preview_limit())
        process_anchors = list(packet.get("process_anchors", []) or [])
        if process_anchors:
            packet["process_anchor_count"] = len(process_anchors)
            packet["process_anchor_labels"] = [
                str(row.get("sa_label", "") or "")
                for row in process_anchors
                if isinstance(row, dict) and str(row.get("sa_label", "") or "")
            ][: self._trace_item_preview_limit()]
            packet["process_anchors"] = self._compact_process_anchor_rows(
                process_anchors,
                limit=self._trace_item_preview_limit(),
            )
        text = str(packet.get("normalized_text", "") or "")
        max_chars = max(32, int(self.config.observability.trace_text_preview_chars))
        if len(text) > max_chars:
            packet["normalized_text_preview"] = text[:max_chars]
            packet["normalized_text_length"] = len(text)
            packet["normalized_text"] = text[:max_chars]
        packet["unit_count"] = len(units)
        return packet

    def _compact_process_anchor_rows(self, rows: list[dict], *, limit: int) -> list[dict]:
        compact = []
        for row in list(rows or [])[: max(1, int(limit))]:
            if not isinstance(row, dict):
                continue
            meta = dict(row.get("anchor_meta", {}) or {})
            compact.append(
                {
                    "sa_label": str(row.get("sa_label", "") or ""),
                    "display_text": str(row.get("display_text", "") or ""),
                    "family": str(row.get("family", "") or ""),
                    "source_type": str(row.get("source_type", "") or ""),
                    "real_energy": float(row.get("real_energy", 0.0) or 0.0),
                    "virtual_energy": float(row.get("virtual_energy", 0.0) or 0.0),
                    "cognitive_pressure": float(row.get("cognitive_pressure", 0.0) or 0.0),
                    "anchor_meta": {
                        "schema_id": str(meta.get("schema_id", "") or ""),
                        "cue_id": str(meta.get("cue_id", "") or ""),
                        "process_anchor_role": str(meta.get("process_anchor_role", "") or ""),
                        "answer_payload_visible": bool(meta.get("answer_payload_visible", False)),
                        "category_label": bool(meta.get("category_label", False)),
                        "hard_reply_route": bool(meta.get("hard_reply_route", False)),
                        "regex_answer_route": bool(meta.get("regex_answer_route", False)),
                        "keyword_hard_gate": bool(meta.get("keyword_hard_gate", False)),
                        "full_sentence_macro": bool(meta.get("full_sentence_macro", False)),
                        "policy": str(meta.get("policy", "") or ""),
                    },
                }
            )
        return compact

    def _summarize_competition(self, competition: dict) -> dict:
        row = dict(competition or {})
        selected = list(row.get("selected_items", []) or [])
        row["selected_items"] = self._compact_rows(selected, limit=self._trace_item_preview_limit())
        row["selected_count"] = len(selected)
        return row

    def _summarize_r_state(self, r_state: dict) -> dict:
        source = dict(r_state or {})
        item_limit = max(1, int(self._trace_r_state_item_preview_limit()))
        heads = []
        total_items = 0
        for head in source.get("heads", []) or []:
            items = list((head or {}).get("items", []) or [])
            total_items += len(items)
            heads.append(
                {
                    "head_id": str((head or {}).get("head_id", "") or ""),
                    "item_count": len(items),
                    "items": self._compact_rows(items, limit=item_limit),
                }
            )
        source["heads"] = heads
        source["total_head_item_count"] = total_items
        return source

    def _compact_rows(self, rows: list[dict], *, limit: int) -> list[dict]:
        compact = []
        for row in list(rows or [])[: max(1, int(limit))]:
            if not isinstance(row, dict):
                continue
            item = {
                "sa_label": str(row.get("sa_label", "") or ""),
                "display_text": str(row.get("display_text", "") or ""),
                "family": str(row.get("family", "") or ""),
                "source_type": str(row.get("source_type", "") or ""),
                "real_energy": float(row.get("real_energy", 0.0) or 0.0),
                "virtual_energy": float(row.get("virtual_energy", 0.0) or 0.0),
                "cognitive_pressure": float(row.get("cognitive_pressure", 0.0) or 0.0),
            }
            for key in (
                "query_weight",
                "attention_score",
                "focus_score",
                "continuation_bonus",
                "successor_bias",
                "emotion_multiplier",
                "focus_order_index",
                "focus_family_bucket",
                "focus_family_budget_relaxed",
                "action_attention_boost",
                "action_attention_suppression",
                "action_attention_net_bias",
            ):
                if key in row:
                    item[key] = row.get(key)
            if "action_attention_sources" in row:
                item["action_attention_sources"] = list(row.get("action_attention_sources", []) or [])[:4]
            if "query_sources" in row:
                item["query_sources"] = list(row.get("query_sources", []) or [])[:4]
            compact.append(item)
        return compact

    def _compact_focus_continuation_trace(self, trace: dict) -> dict:
        row = dict(trace or {})
        row["current_labels"] = list(row.get("current_labels", []) or [])[: self.config.attention.focus_limit]
        row["recent_entries"] = list(row.get("recent_entries", []) or [])[-4:]
        row["replay_candidates"] = list(row.get("replay_candidates", []) or [])[:4]
        return row

    def _compact_bn_rows(self, rows: list[dict]) -> list[dict]:
        return [self._compact_bn_row(row) for row in list(rows or [])]

    def _compact_bn_row(self, row: dict) -> dict:
        matched = dict((row or {}).get("matched_tokens", {}) or {})
        token_limit = max(1, int(self._trace_matched_token_preview_limit()))
        return {
            "memory_id": str((row or {}).get("memory_id", "") or ""),
            "tick_index": int((row or {}).get("tick_index", -1) or -1),
            "memory_kind": str((row or {}).get("memory_kind", "") or ""),
            "score": float((row or {}).get("score", 0.0) or 0.0),
            "normalized_weight": float((row or {}).get("normalized_weight", 0.0) or 0.0),
            "match_efficiency": float((row or {}).get("match_efficiency", 0.0) or 0.0),
            "grasp_confidence": float((row or {}).get("grasp_confidence", 0.0) or 0.0),
            "b_real_energy": float((row or {}).get("b_real_energy", 0.0) or 0.0),
            "b_virtual_energy": float((row or {}).get("b_virtual_energy", 0.0) or 0.0),
            "b_effective_real_energy": float((row or {}).get("b_effective_real_energy", 0.0) or 0.0),
            "b_effective_virtual_energy": float((row or {}).get("b_effective_virtual_energy", 0.0) or 0.0),
            "energy_transfer": dict((row or {}).get("energy_transfer", {}) or {}),
            "source_text": str((row or {}).get("source_text", "") or ""),
            "snapshot_ref": dict((row or {}).get("snapshot_ref", {}) or {}),
            "snapshot_preview": dict((row or {}).get("snapshot_preview", {}) or {}),
            "candidate_sources": list((row or {}).get("candidate_sources", []) or []),
            "matched_tokens": {key: list(value or [])[:token_limit] for key, value in matched.items()},
            "score_breakdown": dict((row or {}).get("score_breakdown", {}) or {}),
            "relative_relation_score": float((row or {}).get("relative_relation_score", 0.0) or 0.0),
            "relative_relation_raw_score": float((row or {}).get("relative_relation_raw_score", 0.0) or 0.0),
            "relation_channels": dict((row or {}).get("relation_channels", {}) or {}),
            "relation_matches": list((row or {}).get("relation_matches", []) or [])[:6],
            "learned_score": float((row or {}).get("learned_score", 0.0) or 0.0),
            "learned_contributions": list((row or {}).get("learned_contributions", []) or [])[:6],
        }

    def _compact_thought_view(self, thought_view: dict) -> dict:
        row = dict(thought_view or {})
        if "fast" in row:
            fast = dict(row.get("fast", {}) or {})
            fast["bn"] = self._compact_bn_rows(fast.get("bn", []))
            row["fast"] = fast
        if "slow" in row:
            slow = dict(row.get("slow", {}) or {})
            slow["bn_prime"] = self._compact_bn_rows(slow.get("bn_prime", []))
            row["slow"] = slow
        focus = dict(row.get("focus_reason", {}) or {})
        if focus:
            focus["ranked_items"] = self._compact_rows(focus.get("ranked_items", []), limit=5)
            row["focus_reason"] = focus
        return row

    def _compact_explainability(self, explainability: dict) -> dict:
        row = dict(explainability or {})
        row["fast_bn"] = [self._compact_bn_reason(item) for item in list(row.get("fast_bn", []) or [])]
        row["slow_bn"] = [self._compact_bn_reason(item) for item in list(row.get("slow_bn", []) or [])]
        focus = dict(row.get("focus", {}) or {})
        if focus:
            focus["ranked_items"] = self._compact_rows(focus.get("ranked_items", []), limit=6)
            row["focus"] = focus
        return row

    def _compact_bn_reason(self, row: dict) -> dict:
        matched = dict((row or {}).get("matched_tokens", {}) or {})
        token_limit = max(1, int(self._trace_matched_token_preview_limit()))
        compact = dict(row or {})
        compact["matched_tokens"] = {key: list(value or [])[:token_limit] for key, value in matched.items()}
        return compact

    def _compact_action_trace(self, action_trace: dict) -> dict:
        row = dict(action_trace or {})
        candidates = [dict(item) for item in list(row.get("candidates", []) or []) if isinstance(item, dict)]
        text_candidates = [
            dict(item)
            for item in candidates
            if str(item.get("action_id", "") or "").startswith("action::text_")
        ]
        row["candidate_count"] = len(candidates)
        row["text_candidate_count"] = len(text_candidates)
        row["candidates"] = candidates[:8]
        row["text_candidate_preview"] = text_candidates[:12]
        return row

    def _compact_innate_traces(self, traces: dict | None) -> dict:
        result = {
            "schema_id": "innate_runtime_trace/v1",
            "enabled": bool(self.config.innate_rules.enabled),
            "validation": self.innate_engine.validate(),
            "actuator_registry": self.innate_engine.actuator_registry(),
            "action_registry": self.innate_engine.action_registry(),
            "phases": {},
            "phase_order": [],
            "total_hit_count": 0,
            "learning_events": [],
        }
        for phase, trace in dict(traces or {}).items():
            row = dict(trace or {})
            compact = {
                "schema_id": row.get("schema_id", "innate_phase_trace/v1"),
                "enabled": bool(row.get("enabled", True)),
                "phase": str(row.get("phase", phase) or phase),
                "rule_count": int(row.get("rule_count", 0) or 0),
                "hit_count": int(row.get("hit_count", 0) or 0),
                "hits": list(row.get("hits", []) or [])[:8],
                "suppressed": list(row.get("suppressed", []) or [])[:6],
                "items": list(row.get("items", []) or [])[:8],
                "action_nodes": list(row.get("action_nodes", []) or [])[:8],
                "action_biases": list(row.get("action_biases", []) or [])[:8],
                "emotion_deltas": dict(row.get("emotion_deltas", {}) or {}),
                "learning_events": list(row.get("learning_events", []) or [])[:8],
                "attention_biases": list(row.get("attention_biases", []) or [])[:6],
                "safety_gate": list(row.get("safety_gate", []) or [])[:6],
                "metrics": dict(row.get("metrics", {}) or {}),
                "fatigue": dict(row.get("fatigue", {}) or {}),
            }
            result["phases"][phase] = compact
            result["phase_order"].append(phase)
            result["total_hit_count"] += int(compact["hit_count"])
            result["learning_events"].extend(list(compact.get("learning_events", []) or []))
        result["learning_events"] = result["learning_events"][:16]
        return result

    def _r_state_to_attention_candidates(self, r_state: dict) -> list[dict]:
        """
        Fixed-budget attention candidate set.

        We intentionally do not scan the full pool here; we reuse the `R_state` heads
        as the bounded candidate set for attention selection.
        """

        merged: dict[str, dict] = {}
        for head in r_state.get("heads", []) or []:
            for row in head.get("items", []) or []:
                label = str((row or {}).get("sa_label", "") or "")
                if not label:
                    continue
                existing = merged.get(label)
                if existing is None:
                    merged[label] = dict(row)
                    continue
                existing["attention_score"] = max(float(existing.get("attention_score", 0.0) or 0.0), float(row.get("attention_score", 0.0) or 0.0))
                existing["query_weight"] = max(float(existing.get("query_weight", 0.0) or 0.0), float(row.get("query_weight", 0.0) or 0.0))
                existing["real_energy"] = max(float(existing.get("real_energy", 0.0) or 0.0), float(row.get("real_energy", 0.0) or 0.0))
                existing["virtual_energy"] = max(float(existing.get("virtual_energy", 0.0) or 0.0), float(row.get("virtual_energy", 0.0) or 0.0))
                current_sources = {
                    str(item or "")
                    for item in list(existing.get("current_source_types", []) or []) + list((row or {}).get("current_source_types", []) or [])
                    if str(item or "")
                }
                if current_sources:
                    existing["current_source_types"] = sorted(current_sources)
                    existing["current_tick_item"] = True
        rows = list(merged.values())
        base_limit = max(self.config.attention.focus_limit * 8, self.config.observability.trace_item_preview_limit * 2, 64)
        attention_budget = self.runtime_budget_controller.attention_candidate_limit(base_limit=base_limit)
        limit = max(self.config.attention.focus_limit, int(attention_budget["limit"]))
        r_state.setdefault("runtime_budget", {})
        r_state["runtime_budget"]["attention_candidate_budget"] = attention_budget
        if len(rows) <= limit:
            rows.sort(key=lambda item: (-float(item.get("attention_score", 0.0) or 0.0), -float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
            return rows
        return nsmallest(
            limit,
            rows,
            key=lambda item: (-float(item.get("attention_score", 0.0) or 0.0), -float(item.get("query_weight", 0.0) or 0.0), str(item.get("sa_label", "") or "")),
        )

    def _shape_focus_family_budget(self, *, ranked_items: list[dict], raw_selected_items: list[dict], action_modulation: dict | None = None) -> tuple[list[dict], dict]:
        """
        Shape the finite focus-memory window without mutating state-pool energy.

        Raw attention still decides the score order. This helper only prevents a
        single SA family from occupying the whole slow-system focus window when
        there are other credible ranked candidates available.
        """

        focus_limit = max(1, int(self.config.attention.focus_limit))
        raw_selected = [dict(item) for item in list(raw_selected_items or []) if isinstance(item, dict)]
        if not bool(getattr(self.config.attention, "focus_family_budget_enabled", True)):
            labels = [str(item.get("sa_label", "") or "") for item in raw_selected if str(item.get("sa_label", "") or "")]
            return raw_selected[:focus_limit], {
                "schema_id": "focus_family_budget_trace/v1",
                "enabled": False,
                "policy": "disabled_raw_attention_window",
                "focus_limit": int(focus_limit),
                "raw_selected_labels": labels[:focus_limit],
                "balanced_selected_labels": labels[:focus_limit],
                "family_counts": {},
                "family_caps": {},
                "overflow_count": 0,
                "overflow_preview": [],
                "relaxed_fill_count": 0,
            }

        ranked = [dict(item) for item in list(ranked_items or []) if isinstance(item, dict)]
        if not ranked:
            ranked = raw_selected
        family_caps = self._modulated_focus_family_caps(self._focus_family_caps(), action_modulation or {})
        selected: list[dict] = []
        selected_labels: set[str] = set()
        family_counts: dict[str, int] = {}
        overflow: list[dict] = []
        overflow_count = 0

        for item in ranked:
            if len(selected) >= focus_limit:
                break
            label = str(item.get("sa_label", "") or "")
            if not label or label in selected_labels:
                continue
            bucket = self._focus_family_bucket(item)
            cap = max(0, int(family_caps.get(bucket, family_caps.get("other", focus_limit)) or 0))
            count = int(family_counts.get(bucket, 0) or 0)
            if count >= cap:
                overflow_count += 1
                if len(overflow) < max(8, focus_limit * 2):
                    overflow.append(self._focus_family_overflow_row(item, bucket, reason="family_cap"))
                continue
            enriched = dict(item)
            enriched["focus_family_bucket"] = bucket
            selected.append(enriched)
            selected_labels.add(label)
            family_counts[bucket] = count + 1

        relaxed_fill_count = 0
        if len(selected) < focus_limit:
            for item in ranked:
                if len(selected) >= focus_limit:
                    break
                label = str(item.get("sa_label", "") or "")
                if not label or label in selected_labels:
                    continue
                bucket = self._focus_family_bucket(item)
                enriched = dict(item)
                enriched["focus_family_bucket"] = bucket
                enriched["focus_family_budget_relaxed"] = True
                selected.append(enriched)
                selected_labels.add(label)
                family_counts[bucket] = int(family_counts.get(bucket, 0) or 0) + 1
                relaxed_fill_count += 1

        raw_labels = [str(item.get("sa_label", "") or "") for item in raw_selected if str(item.get("sa_label", "") or "")]
        balanced_labels = [str(item.get("sa_label", "") or "") for item in selected if str(item.get("sa_label", "") or "")]
        return selected, {
            "schema_id": "focus_family_budget_trace/v1",
            "enabled": True,
            "policy": "ranked_attention_then_family_caps_no_energy_mutation",
            "focus_limit": int(focus_limit),
            "raw_selected_labels": raw_labels[:focus_limit],
            "balanced_selected_labels": balanced_labels,
            "family_counts": {key: int(value) for key, value in sorted(family_counts.items())},
            "family_caps": {key: int(value) for key, value in sorted(family_caps.items())},
            "action_modulation": dict(action_modulation or {}),
            "overflow_count": int(overflow_count),
            "overflow_preview": overflow[: max(1, min(8, focus_limit))],
            "relaxed_fill_count": int(relaxed_fill_count),
            "changed": raw_labels[:focus_limit] != balanced_labels,
        }

    def _focus_family_caps(self) -> dict[str, int]:
        cfg = self.config.attention
        focus_limit = max(1, int(getattr(cfg, "focus_limit", 8) or 8))

        def cap(name: str, fallback: int) -> int:
            return max(0, min(focus_limit, int(getattr(cfg, name, fallback) or 0)))

        return {
            "text": cap("focus_family_text_max", 4),
            "vision": cap("focus_family_vision_max", 3),
            "audio": cap("focus_family_audio_max", 3),
            "cognitive_feeling": cap("focus_family_cognitive_feeling_max", 2),
            "emotion": cap("focus_family_emotion_max", 2),
            "action": cap("focus_family_action_max", 2),
            "time": cap("focus_family_time_max", 1),
            "rhythm": cap("focus_family_rhythm_max", 1),
            "expectation_pressure": cap("focus_family_expectation_pressure_max", 2),
            "other": cap("focus_family_other_max", 2),
        }

    def _modulated_focus_family_caps(self, family_caps: dict[str, int], action_modulation: dict) -> dict[str, int]:
        caps = {str(key): int(value) for key, value in dict(family_caps or {}).items()}
        if not action_modulation:
            return caps
        focus_limit = max(1, int(getattr(self.config.attention, "focus_limit", 8) or 8))
        diversity_gain = max(0.0, float(action_modulation.get("diversity_gain", 0.0) or 0.0))
        release_labels = {str(label or "") for label in list(action_modulation.get("release_labels", []) or []) if str(label or "")}
        if diversity_gain > 0.0:
            for key in ("text", "vision", "audio", "other", "cognitive_feeling", "expectation_pressure"):
                caps[key] = min(focus_limit, max(int(caps.get(key, 0) or 0), int(caps.get(key, 0) or 0) + 1))
        if release_labels:
            # A release action is a short-lived anti-lock control. It should not
            # delete the old family, only prevent it from filling the whole
            # finite focus window while the system tries another angle.
            release_buckets = {self._focus_family_bucket({"sa_label": label}) for label in release_labels}
            for bucket in release_buckets:
                if bucket in caps:
                    caps[bucket] = max(1, min(int(caps[bucket]), max(1, focus_limit - 1)))
        return caps

    def _focus_family_overflow_row(self, item: dict, bucket: str, *, reason: str) -> dict:
        return {
            "sa_label": str((item or {}).get("sa_label", "") or ""),
            "display_text": str((item or {}).get("display_text", "") or ""),
            "focus_family_bucket": str(bucket or "other"),
            "focus_score": float((item or {}).get("focus_score", 0.0) or 0.0),
            "real_energy": float((item or {}).get("real_energy", 0.0) or 0.0),
            "virtual_energy": float((item or {}).get("virtual_energy", 0.0) or 0.0),
            "reason": str(reason or "family_cap"),
        }

    def _focus_family_bucket(self, item: dict) -> str:
        label = str((item or {}).get("sa_label", "") or "").lower()
        family = str((item or {}).get("family", "") or "").lower()
        source_type = str((item or {}).get("source_type", "") or "").lower()
        prefixes = (label, family, source_type)
        if label.startswith("expectation_pressure::") or "expectation_pressure" in prefixes:
            return "expectation_pressure"
        if label.startswith("action::") or label.startswith("action_feedback::") or label.startswith("text_action::"):
            return "action"
        if family in {"action", "action_feedback", "text_action"} or source_type in {"action", "action_feedback", "text_action"}:
            return "action"
        if label.startswith("emotion::") or family == "emotion" or source_type == "emotion":
            return "emotion"
        if label.startswith("feeling::") or family == "cognitive_feeling" or source_type == "cognitive_feeling":
            return "cognitive_feeling"
        if label.startswith("timefelt::") or family in {"time", "time_feeling"} or source_type in {"time", "time_feeling"}:
            return "time"
        if label.startswith("rhythmfelt::") or family == "rhythm" or source_type == "rhythm":
            return "rhythm"
        if label.startswith(("audio::", "sound::", "hearing::")) or family.startswith(("audio", "hearing")) or source_type.startswith(("audio", "hearing")):
            return "audio"
        if label.startswith(("vision::", "visual::", "image::")) or family.startswith(("vision", "visual", "image")) or source_type.startswith(("vision", "visual", "image")):
            return "vision"
        if label.startswith(("text::", "phrase::")) or family in {"text", "learned_text_phrase", "text_phrase"} or source_type == "external_text":
            return "text"
        return "other"

    def _stabilize_focus_order(self, selected_items: list[dict]) -> tuple[list[dict], dict]:
        """
        Keep attention winners, but order the focus window for slow-system memory.

        Attention decides *which* objects enter focus. This helper only decides
        how ordered external evidence is written into focus memory, so text
        sequence statistics do not depend on score tie-breaks.
        """

        rows = []
        for rank, item in enumerate(selected_items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            row = dict(item)
            rows.append(
                {
                    "rank": int(rank),
                    "item": row,
                    "label": label,
                    "is_orderable_text": self._is_orderable_text_focus_item(row),
                    "source_key": self._focus_source_order_key(row, default_rank=rank),
                }
            )
        if not rows:
            return [], {
                "schema_id": "focus_order_trace/v1",
                "policy": "empty",
                "raw_attention_labels": [],
                "ordered_focus_labels": [],
                "text_ordered_count": 0,
                "non_text_count": 0,
                "changed": False,
            }

        orderable_count = sum(1 for row in rows if bool(row.get("is_orderable_text", False)))
        should_reorder_text = orderable_count >= 2
        if should_reorder_text:
            ordered_text_rows = sorted(
                [row for row in rows if bool(row.get("is_orderable_text", False))],
                key=lambda row: (
                    row.get("source_key", (0, int(row.get("rank", 0) or 0))),
                    int(row.get("rank", 0) or 0),
                ),
            )
            text_iter = iter(ordered_text_rows)
            ordered_rows = [next(text_iter) if bool(row.get("is_orderable_text", False)) else row for row in rows]
            policy = "orderable_external_text_relative_order_only"
        else:
            ordered_rows = list(rows)
            policy = "attention_rank_preserved_no_comparable_text_order"

        ordered_items = [dict(row["item"]) for row in ordered_rows]
        raw_labels = [str(row.get("label", "") or "") for row in rows]
        ordered_labels = [str(row.get("label", "") or "") for row in ordered_rows]
        for order_index, item in enumerate(ordered_items):
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            anchor_meta["focus_order_index"] = int(order_index)
            item["anchor_meta"] = anchor_meta
            item["focus_order_index"] = int(order_index)
        return ordered_items, {
            "schema_id": "focus_order_trace/v1",
            "policy": policy,
            "raw_attention_labels": raw_labels,
            "ordered_focus_labels": ordered_labels,
            "text_ordered_count": int(orderable_count),
            "non_text_count": int(len(rows) - orderable_count),
            "changed": raw_labels != ordered_labels,
        }

    def _is_orderable_text_focus_item(self, item: dict) -> bool:
        label = str((item or {}).get("sa_label", "") or "")
        if not label.startswith(("text::", "phrase::")):
            return False
        source_type = str((item or {}).get("source_type", "") or "")
        family = str((item or {}).get("family", "") or "")
        return source_type == "external_text" or family in {"text", "learned_text_phrase", "text_phrase"}

    def _focus_source_order_key(self, item: dict, *, default_rank: int) -> tuple[int, int, int]:
        anchor_meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        tick_value = (item or {}).get("last_seen_tick", (item or {}).get("tick_index", anchor_meta.get("tick_index", self.tick_index)))
        position_value = (item or {}).get("position", anchor_meta.get("position", default_rank))
        try:
            tick_key = int(tick_value)
        except (TypeError, ValueError):
            tick_key = int(self.tick_index)
        try:
            position_key = int(position_value)
        except (TypeError, ValueError):
            position_key = int(default_rank)
        return (tick_key, position_key, int(default_rank))

    def _append_r_state_head(self, r_state: dict, head_id: str, rows: list[dict]) -> dict:
        updated = dict(r_state or {})
        heads = list(updated.get("heads", []) or [])
        if rows:
            heads.append({"head_id": str(head_id or "head_incremental"), "items": list(rows)})
        updated["heads"] = heads
        updated["head_count"] = len(heads)
        available = list(updated.get("available_head_ids", []) or [])
        if head_id not in available:
            available.append(str(head_id or "head_incremental"))
        updated["available_head_ids"] = available
        preview = list(updated.get("merged_preview", []) or [])
        seen = {str(label or "") for label in preview if str(label or "")}
        for row in rows:
            label = str((row or {}).get("sa_label", "") or "")
            if label and label not in seen:
                seen.add(label)
                preview.append(label)
        updated["merged_preview"] = preview
        return updated

    def _should_rerun_timefelt_recall(self, time_trace: dict) -> bool:
        dominant = dict((time_trace or {}).get("dominant_peak", {}) or {})
        confidence = float(dominant.get("confidence", 0.0) or 0.0)
        max_energy = max([float(item.get("real_energy", 0.0) or 0.0) for item in (time_trace or {}).get("items", []) or []] or [0.0])
        return (
            confidence >= float(self.config.time_feeling.rerun_recall_confidence_threshold)
            and max_energy >= float(self.config.time_feeling.rerun_recall_energy_threshold)
        )

    def _ingest_text(self, text: str) -> tuple[dict, dict, list[dict]]:
        input_packet = self.text_sensor.ingest(text, tick_index=self.tick_index)
        self.sa_registry.observe_sequence(input_packet["units"])
        cache_key = (
            str(input_packet.get("normalized_text", "") or ""),
            str(input_packet.get("source_type", "") or ""),
            int(self.config.text_sensor.competition_limit),
            int(self.config.text_sensor.budget_limit),
        )
        cached = self._text_ingest_cache.get(cache_key)
        if cached is not None:
            selected_template = list(cached.get("selected_items", []) or [])
            competition = {
                "selected_items": [dict(item) for item in selected_template],
                "cache": {"hit": True, "kind": "text_competition_template"},
            }
        else:
            competition = self.sa_registry.compete(
                input_packet["units"],
                source_type=input_packet["source_type"],
                max_items=self.config.text_sensor.competition_limit,
            )
            selected_template = [dict(item) for item in list(competition.get("selected_items", []) or [])]
            self._text_ingest_cache[cache_key] = {"selected_items": selected_template}
            if len(self._text_ingest_cache) > 8:
                first_key = next(iter(self._text_ingest_cache))
                self._text_ingest_cache.pop(first_key, None)
            competition = dict(competition)
            competition["cache"] = {"hit": False, "kind": "text_competition_template"}
        external_items = list(competition.get("selected_items", []) or [])
        process_anchors = self._dialogue_turn_task_anchors(input_packet, external_items)
        process_anchors.extend(self._dialogue_input_process_anchors(input_packet, external_items))
        if process_anchors:
            input_packet["process_anchors"] = [dict(item) for item in process_anchors]
            input_packet["process_anchor_labels"] = [
                str(item.get("sa_label", "") or "")
                for item in process_anchors
                if str(item.get("sa_label", "") or "")
            ]
        external_items.extend(process_anchors)
        self._remember_dialogue_input(str(input_packet.get("normalized_text", "") or ""))
        return input_packet, competition, external_items

    def _dialogue_turn_task_anchors(self, input_packet: dict, text_items: list[dict]) -> list[dict]:
        normalized = str((input_packet or {}).get("normalized_text", "") or "").strip()
        if not normalized:
            return []
        text_labels = [
            str(item.get("sa_label", "") or "")
            for item in list(text_items or [])[:12]
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        ]
        length = len(normalized)
        urgency = 0.42 + min(0.22, length / 120.0)
        if any(mark in normalized for mark in ("?", "？")):
            urgency += 0.10
        if length <= 8:
            urgency += 0.04
        urgency = round(min(0.78, urgency), 4)
        meta = {
            "schema_id": "dialogue_turn_closure_anchor/v1",
            "source": "external_text_turn",
            "turn_text_preview": normalized[:80],
            "target_labels": ["text_action::draft_state", "action::text_commit"],
            "target_text": "",
            "strictness": 0.18,
            "reply_closure_need": urgency,
            "external_text_labels": text_labels,
            "policy": "soft_dialogue_closure_anchor_not_reply_content_or_keyword_route",
        }
        return [
            {
                "sa_label": "task::reply_to_current_user_turn",
                "display_text": "需要回应当前用户输入",
                "family": "task",
                "source_type": "task_anchor",
                "real_energy": urgency,
                "virtual_energy": 0.08,
                "cognitive_pressure": round(urgency * 0.35, 4),
                "anchor_meta": dict(meta),
            },
            {
                "sa_label": "intention::dialogue_turn_closure",
                "display_text": "本轮对话需要形成闭合",
                "family": "intention",
                "source_type": "intention_anchor",
                "real_energy": round(max(0.30, urgency - 0.08), 4),
                "virtual_energy": 0.06,
                "cognitive_pressure": round(urgency * 0.24, 4),
                "anchor_meta": {**meta, "strictness": 0.12},
            },
        ]

    def _remember_dialogue_input(self, normalized: str) -> None:
        text = str(normalized or "").strip()
        if not text:
            return
        self._recent_dialogue_inputs.append(text)
        self._recent_dialogue_inputs = self._recent_dialogue_inputs[-8:]

    def _dialogue_process_anchor_item(
        self,
        label: str,
        *,
        real: float,
        virtual: float = 0.05,
        display_text: str = "",
        cue_id: str,
        evidence: dict,
        text_preview: str,
        text_labels: list[str],
    ) -> dict:
        family = str(label).split("::", 1)[0] if "::" in str(label) else "dialogue_process"
        strength = max(0.0, min(0.92, float(real or 0.0)))
        v_energy = max(0.0, min(strength, float(virtual or 0.0)))
        return {
            "sa_label": str(label),
            "display_text": str(display_text or label),
            "family": family,
            "source_type": "dialogue_process_anchor",
            "real_energy": _round4(strength),
            "virtual_energy": _round4(v_energy),
            "cognitive_pressure": _round4(max(0.0, strength - v_energy) * 0.32),
            "anchor_meta": {
                "schema_id": "dialogue_input_process_anchor/v1",
                "source": "external_text_turn",
                "process_anchor_role": "input_shape_and_process_pressure",
                "cue_id": str(cue_id),
                "evidence_features": dict(evidence or {}),
                "turn_text_preview": str(text_preview or "")[:80],
                "external_text_labels": list(text_labels or [])[:12],
                "answer_payload_visible": False,
                "category_label": False,
                "reply_text": "",
                "hard_reply_route": False,
                "regex_answer_route": False,
                "keyword_hard_gate": False,
                "full_sentence_macro": False,
                "external_surface_hard_gate": False,
                "used_in_strict_teacher_off_input": True,
                "policy": "soft_state_field_process_anchor_not_answer_or_reply_router",
            },
        }

    def _dialogue_input_process_anchors(self, input_packet: dict, text_items: list[dict]) -> list[dict]:
        """
        Build legal process anchors from the current input shape.

        These rows restore the old controlled Fresh300 style of state-field
        context: low-grain process pressure, short-term slots, and focusable
        cues. They do not contain final replies or answer-family labels.
        """

        normalized = str((input_packet or {}).get("normalized_text", "") or "").strip()
        if not normalized:
            return []
        text_labels = [
            str(item.get("sa_label", "") or "")
            for item in list(text_items or [])[:12]
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        ]
        length = len(normalized)
        cjk_or_alnum = sum(1 for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff")
        symbol_count = sum(
            1
            for char in normalized
            if not char.isspace() and not char.isalnum() and not ("\u4e00" <= char <= "\u9fff")
        )
        symbol_ratio = symbol_count / max(1, len(normalized))
        digit_count = sum(1 for char in normalized if char.isdigit())
        questionish = "?" in normalized or "？" in normalized
        repeated_recently = normalized in self._recent_dialogue_inputs[-4:]
        rows: list[dict] = []
        seen: set[str] = set()

        def has_any(tokens: tuple[str, ...]) -> bool:
            return any(token and token in normalized for token in tokens)

        def add(label: str, *, real: float, cue_id: str, evidence: dict | None = None, display_text: str = "") -> None:
            if label in seen:
                return
            seen.add(label)
            rows.append(
                self._dialogue_process_anchor_item(
                    label,
                    real=real,
                    display_text=display_text,
                    cue_id=cue_id,
                    evidence={
                        "length": length,
                        "symbol_ratio": _round4(symbol_ratio),
                        "digit_count": digit_count,
                        "questionish": questionish,
                        "repeated_recently": repeated_recently,
                        **dict(evidence or {}),
                    },
                    text_preview=normalized,
                    text_labels=text_labels,
                )
            )

        add("state::current_input_new", real=0.34, cue_id="current_turn_present", display_text="current input is new")
        add("goal::understand_current_task", real=0.32, cue_id="current_turn_present", display_text="understand current turn")
        if questionish:
            add("dialogue_process::question_form", real=0.48, cue_id="question_shape", display_text="question-shaped input")
            add("short_term_slot::current_question", real=0.42, cue_id="question_shape", display_text="current question slot")
        if length <= 10 and cjk_or_alnum > 0 and symbol_ratio < 0.35:
            add("dialogue_process::brief_turn", real=0.38, cue_id="short_clean_turn", display_text="brief clean turn")
            add("state::moderate_grasp", real=0.34, cue_id="short_clean_turn", display_text="moderate grasp")
        if has_any(("你好", "在吗", "早安", "晚上好", "哈喽", "嗨", "hi", "hello")):
            add("dialogue_process::social_opening", real=0.46, cue_id="social_opening_shape", display_text="social opening")
            add("short_term_slot::simple_answer", real=0.36, cue_id="social_opening_shape", display_text="simple answer slot")
        if has_any(("你是谁", "介绍一下", "你能做什么", "不能做什么", "你现在算", "你是什么")):
            add("dialogue_process::self_identity_question", real=0.46, cue_id="identity_or_boundary_question", display_text="identity boundary question")
            add("state::capability_boundary_needed", real=0.42, cue_id="identity_or_boundary_question", display_text="capability boundary needed")
            add("short_term_slot::self_description", real=0.34, cue_id="identity_or_boundary_question", display_text="self description slot")
        if has_any(("不知道", "不确定", "没见过", "陌生", "定义", "什么意思", "是什么", "教我", "你可以教", "先教")):
            add("unknown::knowledge_gap", real=0.52, cue_id="unknown_or_learning_shape", display_text="knowledge gap")
            add("state::low_grasp", real=0.44, cue_id="unknown_or_learning_shape", display_text="low grasp")
            add("state::ask_teaching_available", real=0.42, cue_id="unknown_or_learning_shape", display_text="teaching can be requested")
            add("short_term_slot::learning_intent", real=0.38, cue_id="unknown_or_learning_shape", display_text="learning intent slot")
        if has_any(("我教你", "记住", "记一下", "以后我说", "设定", "我来教", "更正", "纠正")):
            add("teacher_loop::multi_turn_learning", real=0.50, cue_id="user_teaching_shape", display_text="user teaching loop")
            add("goal::store_new_experience", real=0.46, cue_id="user_teaching_shape", display_text="store new experience")
            add("short_term_slot::learning_intent", real=0.40, cue_id="user_teaching_shape", display_text="learning intent slot")
        if has_any(("不对", "改一下", "纠正", "说错", "不是", "而是", "最新", "按新的")):
            add("cue::latest_correction", real=0.54, cue_id="correction_shape", display_text="latest correction cue")
            add("state::contradiction_detected", real=0.48, cue_id="correction_shape", display_text="contradiction detected")
            add("state::current_question_priority", real=0.42, cue_id="correction_shape", display_text="current turn priority")
            add("short_term_slot::corrected_fact", real=0.42, cue_id="correction_shape", display_text="corrected fact slot")
        if has_any(("难过", "累", "烦", "害怕", "紧张", "焦虑", "委屈", "崩溃", "压力", "陪我", "安静", "不想听大道理")):
            add("emotion::sadness_or_tiredness", real=0.52, cue_id="emotion_disclosure_shape", display_text="emotion disclosure")
            add("emotion::calm_settling", real=0.44, cue_id="emotion_disclosure_shape", display_text="calm settling")
            add("state::task_pressure_low", real=0.40, cue_id="emotion_disclosure_shape", display_text="lower task pressure")
            add("short_term_slot::companion_mode", real=0.40, cue_id="emotion_disclosure_shape", display_text="companion mode slot")
        if has_any(("帮我", "处理", "整理", "计划", "检查", "写点", "做一下", "弄好", "这个", "那个")):
            add("unknown::ambiguous_reference", real=0.44, cue_id="task_clarification_shape", display_text="ambiguous reference")
            add("short_term_slot::need_clarification", real=0.40, cue_id="task_clarification_shape", display_text="need clarification slot")
        if has_any(("桌面", "点击", "按钮", "发送", "删除", "清空", "保存", "打开", "文件", "屏幕", "读回", "控制电脑", "窗口")):
            add("pet_task::desktop_assist", real=0.48, cue_id="desktop_action_shape", display_text="desktop assist")
            add("state::permission_uncertain", real=0.44, cue_id="desktop_action_shape", display_text="permission uncertain")
            add("goal::plan_small_steps", real=0.38, cue_id="desktop_action_shape", display_text="plan small steps")
            add("short_term_slot::safe_next_step", real=0.40, cue_id="desktop_action_shape", display_text="safe next step slot")
        if has_any(("直接删除", "直接发送", "不用确认", "付款", "清空", "覆盖", "危险", "高风险")):
            add("state::high_stakes_or_destructive", real=0.56, cue_id="risky_action_shape", display_text="high stakes or destructive")
            add("future_feedback::bad_if_wrong_action", real=0.48, cue_id="risky_action_shape", display_text="bad if wrong action")
            add("goal::avoid_wrong_action", real=0.42, cue_id="risky_action_shape", display_text="avoid wrong action")
        if digit_count >= 2 or has_any(("加", "减", "乘", "除", "每", "一共", "平均", "剩", "多少", "几", "列式", "公式", "题型")):
            add("knowledge_atom::basic_math_science", real=0.50, cue_id="quantity_relation_shape", display_text="basic math/science relation")
            add("dialogue_process::quantity_relation", real=0.48, cue_id="quantity_relation_shape", display_text="quantity relation")
            add("operation::arithmetic_candidate", real=0.44, cue_id="quantity_relation_shape", display_text="arithmetic candidate")
            add("short_term_slot::calculation_task", real=0.40, cue_id="quantity_relation_shape", display_text="calculation task slot")
            add("goal::reread_before_try", real=0.34, cue_id="quantity_relation_shape", display_text="reread before trying")
        if symbol_ratio >= 0.24 or has_any(("@@@", "###", "%%", "乱", "噪声", "听不清", "看不清", "不清楚", "重发")):
            add("unknown::unparseable_input", real=0.54, cue_id="noise_or_low_coherence_shape", display_text="unparseable input")
            add("state::low_coherence", real=0.50, cue_id="noise_or_low_coherence_shape", display_text="low coherence")
            add("state::no_garbled_output", real=0.42, cue_id="noise_or_low_coherence_shape", display_text="avoid garbled output")
            add("short_term_slot::need_restate", real=0.42, cue_id="noise_or_low_coherence_shape", display_text="need restate slot")
        if repeated_recently or has_any(("重复", "再说一遍", "原话", "连续", "哈哈哈", "好好好", "别当错")):
            add("feeling::external_repetition", real=0.50, cue_id="repetition_shape", display_text="external repetition")
            add("state::still_missing_information", real=0.38, cue_id="repetition_shape", display_text="still missing information")
            add("short_term_slot::repeated_question", real=0.38, cue_id="repetition_shape", display_text="repeated question slot")
            if has_any(("原话", "再说一遍", "别当错", "不是错误")):
                add("feeling::intentional_repeat_ok", real=0.46, cue_id="intentional_repeat_shape", display_text="intentional repeat ok")
        if has_any(("刚刚", "刚才", "最近", "前几天", "好久", "从未", "每天", "到点", "睡前", "明天", "昨天", "现在又")):
            add("timefelt::elapsed", real=0.44, cue_id="time_interval_shape", display_text="time interval felt")
            if has_any(("刚刚", "刚才", "刚做", "刚发生")):
                add("time::relative_interval::just_now", real=0.48, cue_id="recent_time_shape", display_text="just now")
                add("task_state::recent_completion", real=0.38, cue_id="recent_time_shape", display_text="recent completion")
            if has_any(("每天", "到点", "现在又", "睡前")):
                add("time::recurrence::cycle_due", real=0.48, cue_id="recurrence_time_shape", display_text="cycle due")
                add("future_reward::task_progress", real=0.40, cue_id="recurrence_time_shape", display_text="task progress reward")
        if has_any(("之前想", "上次想", "没机会", "不能做", "没法做", "现在可以", "机会", "想起来", "路过")):
            add("intention::wanted_but_blocked", real=0.50, cue_id="deferred_intention_shape", display_text="wanted but blocked")
            add("cue::opportunity_present", real=0.46, cue_id="deferred_intention_shape", display_text="opportunity present")
            add("feeling::unresolved_intention", real=0.40, cue_id="deferred_intention_shape", display_text="unresolved intention")
        if has_any(("组合", "学过", "没见过这题", "A+B", "A+B+C", "公式", "题型", "迁移")):
            add("goal::try_composition", real=0.46, cue_id="composition_transfer_shape", display_text="try composition")
            add("short_term_slot::composition_attempt", real=0.40, cue_id="composition_transfer_shape", display_text="composition attempt slot")
            add("state::ab_composition_ready", real=0.38, cue_id="composition_transfer_shape", display_text="A+B composition ready")
            if has_any(("A+B+C", "三个", "三步")):
                add("state::abc_composition_ready", real=0.38, cue_id="composition_transfer_shape", display_text="A+B+C composition ready")
        if length >= 80 or has_any(("换个话题", "接下来", "刚才先", "现在改", "长话短说")):
            add("state::long_context", real=0.42, cue_id="long_context_or_switch_shape", display_text="long context")
            add("state::topic_switch_detected", real=0.38, cue_id="long_context_or_switch_shape", display_text="topic switch detected")
            add("short_term_slot::current_goal", real=0.38, cue_id="long_context_or_switch_shape", display_text="current goal slot")
        if has_any(("草稿", "回读", "发送前", "你好你", "删掉", "修改最后")):
            add("feeling::self_expression_repetition", real=0.48, cue_id="draft_revision_shape", display_text="self expression repetition")
            add("goal::revise_before_commit", real=0.44, cue_id="draft_revision_shape", display_text="revise before commit")
            add("short_term_slot::draft_revision", real=0.40, cue_id="draft_revision_shape", display_text="draft revision slot")

        return rows[:28]

    def _ingest_multimodal(self, *, text: str, image_bytes: bytes | None, audio_bytes: bytes | None) -> tuple[dict, dict, list[dict], dict]:
        input_packet, competition, external_items = self._ingest_text(text)
        multimodal_trace = {
            "inner_vision": {},
            "inner_audio": {},
            "asset_refs": [],
            "ingested_modalities": [],
        }
        if image_bytes:
            vision_trace = self._ingest_vision_bytes(image_bytes)
            vision_trace = self._register_vision_assets(vision_trace, raw_image_bytes=image_bytes)
            vision_trace = self._strip_raw_multimodal_preview_payloads(vision_trace, modality="vision")
            external_items.extend(vision_trace["state_items"])
            multimodal_trace["inner_vision"] = vision_trace["inner_vision"]
            multimodal_trace["asset_refs"].extend(list(vision_trace.get("asset_refs", []) or []))
            multimodal_trace["ingested_modalities"].append("vision")
        if audio_bytes:
            audio_trace = self._ingest_audio_bytes(audio_bytes)
            audio_trace = self._register_audio_assets(audio_trace, raw_audio_bytes=audio_bytes)
            audio_trace = self._strip_raw_multimodal_preview_payloads(audio_trace, modality="audio")
            external_items.extend(audio_trace["state_items"])
            multimodal_trace["inner_audio"] = audio_trace["inner_audio"]
            multimodal_trace["asset_refs"].extend(list(audio_trace.get("asset_refs", []) or []))
            multimodal_trace["ingested_modalities"].append("audio")
        if text:
            multimodal_trace["ingested_modalities"].append("text")
        multimodal_trace["asset_refs"] = self._dedupe_asset_refs(multimodal_trace.get("asset_refs", []))
        if multimodal_trace["asset_refs"]:
            multimodal_trace["asset_store"] = self.asset_store.summary()
        return input_packet, competition, external_items, multimodal_trace

    def _strip_raw_multimodal_preview_payloads(self, trace: dict, *, modality: str) -> dict:
        result = dict(trace or {})
        if modality == "vision":
            inner = dict(result.get("inner_vision", {}) or {})
            current_frame = dict(inner.get("current_frame", {}) or {})
            current_frame.pop("preview_png_b64", None)
            current_frame.setdefault("reconstruction_basis", "state_pool_numeric_channels")
            current_frame["raw_preview_payload"] = False
            inner["current_frame"] = current_frame
            if not bool(self.config.multimodal_assets.enabled):
                inner.pop("asset_refs", None)
                for key in ("asset_ref", "raw_asset_ref"):
                    current_frame.pop(key, None)
                inner["current_frame"] = current_frame
                objects = []
                for obj in list(inner.get("object_reconstruction", []) or []):
                    if not isinstance(obj, dict):
                        continue
                    row = dict(obj)
                    row.pop("asset_ref", None)
                    row.pop("focus_tile_asset_ref", None)
                    objects.append(row)
                inner["object_reconstruction"] = objects
                result["asset_refs"] = []
            result["inner_vision"] = inner
            return result
        if modality == "audio":
            inner = dict(result.get("inner_audio", {}) or {})
            preview = dict(inner.get("preview_asset_ref", {}) or {})
            preview.pop("preview_wav_b64", None)
            preview.pop("proxy_preview_wav_b64", None)
            preview.setdefault("reconstruction_basis", "state_pool_numeric_channels")
            preview["raw_preview_payload"] = False
            if not bool(self.config.multimodal_assets.enabled):
                for key in ("asset_ref", "feature_asset_ref", "focus_window_asset_ref"):
                    preview.pop(key, None)
                inner.pop("asset_refs", None)
                result["asset_refs"] = []
            inner["preview_asset_ref"] = preview
            result["inner_audio"] = inner
            return result
        return result

    def _ingest_vision_bytes(self, image_bytes: bytes) -> dict:
        mode = str(self.config.vision_sensor.mode or "native_numeric").strip().lower()
        if mode in {"native", "native_numeric", "numeric"}:
            try:
                trace = self.vision_sensor.ingest_image_bytes(
                    image_bytes,
                    tick_index=self.tick_index,
                    focus_state=self.visual_gaze_actuator.state(),
                )
                trace.setdefault("packet", {})["sensor_mode"] = "native_numeric"
                return trace
            except Exception as exc:
                if not self.config.vision_sensor.fallback_to_legacy:
                    raise
                legacy = self.vision_bridge.ingest_image_bytes(image_bytes, tick_index=self.tick_index)
                legacy.setdefault("packet", {})["sensor_mode"] = "legacy_fallback"
                legacy.setdefault("inner_vision", {})["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                return legacy
        legacy = self.vision_bridge.ingest_image_bytes(image_bytes, tick_index=self.tick_index)
        legacy.setdefault("packet", {})["sensor_mode"] = "legacy_bridge"
        return legacy

    def _ingest_audio_bytes(self, audio_bytes: bytes) -> dict:
        mode = str(self.config.audio_sensor.mode or "native_numeric").strip().lower()
        if mode in {"native", "native_numeric", "numeric"}:
            try:
                trace = self.audio_sensor.ingest_wav_bytes(
                    audio_bytes,
                    tick_index=self.tick_index,
                    focus_state=self.auditory_band_actuator.state(),
                )
                trace.setdefault("packet", {})["sensor_mode"] = "native_numeric"
                return trace
            except Exception as exc:
                if not self.config.audio_sensor.fallback_to_legacy:
                    raise
                legacy = self.audio_bridge.ingest_wav_bytes(audio_bytes, tick_index=self.tick_index)
                legacy.setdefault("packet", {})["sensor_mode"] = "legacy_fallback"
                legacy.setdefault("inner_audio", {})["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                return legacy
        legacy = self.audio_bridge.ingest_wav_bytes(audio_bytes, tick_index=self.tick_index)
        legacy.setdefault("packet", {})["sensor_mode"] = "legacy_bridge"
        return legacy

    def _register_vision_assets(self, trace: dict, *, raw_image_bytes: bytes | None = None) -> dict:
        # APV2.1 inner replay must be reconstructed from state-pool numeric SA
        # channels. Raw/near-raw visual assets are intentionally hard-disabled so
        # future observatory work cannot fall back to replaying input media.
        return trace

    def _register_audio_assets(self, trace: dict, *, raw_audio_bytes: bytes | None = None) -> dict:
        # APV2.1 inner replay must be reconstructed from state-pool numeric SA
        # channels. Raw/near-raw audio assets are intentionally hard-disabled so
        # future observatory work cannot fall back to replaying input media.
        return trace

    def _dedupe_asset_refs(self, refs: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for ref in refs or []:
            if not isinstance(ref, dict):
                continue
            asset_id = str(ref.get("asset_id", "") or "")
            if not asset_id or asset_id in seen:
                continue
            seen.add(asset_id)
            rows.append(dict(ref))
        return rows

    def _apply_external_or_bootstrap(self, external_items: list[dict], *, memory_bootstrap: bool) -> None:
        if external_items:
            self.state_pool.apply_external_items(external_items, tick_index=self.tick_index)
            return
        if memory_bootstrap and self.config.allow_memory_bootstrap:
            latest = self.memory.latest_snapshot("state")
            if latest is not None:
                self.state_pool.apply_memory_bootstrap(latest, tick_index=self.tick_index)

    def _consume_pending_action_feedback(self) -> dict:
        pending = dict(self._pending_action_feedback or {})
        self._pending_action_feedback = None
        if not pending:
            queued_feedback = self._consume_queued_external_feedback()
            if queued_feedback:
                feedback_items = self._build_explicit_feedback_items(queued_feedback)
                structured_events = self._structured_action_outcome_events(
                    selected_actions=[],
                    observed_feedback=queued_feedback,
                    planner_feedback={},
                    parameter_events=[],
                )
                return {
                    "applied": True,
                    "selected_actions": [],
                    "observed_feedback": queued_feedback,
                    "planner_feedback": {},
                    "causal_window": {},
                    "feedback_items": feedback_items,
                    "structured_learning_events": structured_events,
                    "source": "external_feedback_queue",
                }
            return {"applied": False, "selected_actions": [], "observed_feedback": {}, "feedback_items": [], "structured_learning_events": []}
        selected_actions = list(pending.get("selected_actions", []) or [])
        feedback_context = dict(pending.get("feedback_context", {}) or {})
        causal_window = dict(pending.get("causal_window", {}) or {})
        observed_feedback = self._observe_action_feedback(selected_actions=selected_actions, feedback_context=feedback_context)
        queued_feedback = self._consume_queued_external_feedback()
        if queued_feedback:
            observed_feedback = self._merge_observed_feedback(observed_feedback, queued_feedback)
        text_feedback_binding = self.text_actuator.apply_feedback_to_recent_action(
            observed_feedback,
            causal_window=causal_window,
        )
        parameter_events = (
            list(causal_window.get("visual_gaze_events", []) or [])
            + list(causal_window.get("auditory_band_events", []) or [])
            + list(causal_window.get("text_parameter_events", []) or [])
        )
        if text_feedback_binding.get("applied"):
            parameter_events.append(
                {
                    "schema_id": "text_action_feedback_binding/v1",
                    "action_id": "action::text_feedback_binding",
                    "parameter_kind": "text_feedback_binding",
                    "target_token": str(text_feedback_binding.get("target_token", "") or ""),
                    "feedback_outcome": str(text_feedback_binding.get("feedback_outcome", "") or ""),
                    "feedback_reference_token": str(text_feedback_binding.get("feedback_reference_token", "") or ""),
                    "feedback_mismatch_basis": str(text_feedback_binding.get("mismatch_basis", "") or ""),
                    "observed_feedback": dict(text_feedback_binding.get("observed_feedback", {}) or {}),
                }
            )
        planner_feedback = self.action_planner.record_feedback(
            selected_actions=selected_actions,
            observed_feedback=observed_feedback,
            parameter_events=parameter_events,
        )
        feedback_items = self._build_action_feedback_items(
            selected_actions=selected_actions,
            observed_feedback=observed_feedback,
            planner_feedback=planner_feedback,
            causal_window=causal_window,
        )
        structured_events = self._structured_action_outcome_events(
            selected_actions=selected_actions,
            observed_feedback=observed_feedback,
            planner_feedback=planner_feedback,
            parameter_events=parameter_events,
        )
        if queued_feedback:
            feedback_items.extend(self._build_explicit_feedback_items(queued_feedback))
        return {
            "applied": True,
            "selected_actions": selected_actions,
            "observed_feedback": observed_feedback,
            "planner_feedback": planner_feedback,
            "causal_window": causal_window,
            "feedback_items": feedback_items,
            "structured_learning_events": structured_events,
            "external_feedback": dict(queued_feedback or {}),
            "text_feedback_binding": text_feedback_binding,
        }

    def _consume_queued_external_feedback(self) -> dict:
        feedback = dict(self._queued_external_feedback or {})
        self._queued_external_feedback = None
        return feedback

    def _merge_queued_external_feedback(self, feedback: dict) -> None:
        if not feedback:
            return
        existing = dict(self._queued_external_feedback or {})
        if not existing:
            self._queued_external_feedback = dict(feedback)
            return
        self._queued_external_feedback = self._merge_observed_feedback(existing, feedback)

    def _merge_observed_feedback(self, observed_feedback: dict, external_feedback: dict) -> dict:
        observed = dict(observed_feedback or {})
        external = dict(external_feedback or {})
        observed_reward = max(0.0, float(observed.get("reward", 0.0) or 0.0))
        observed_punishment = max(0.0, float(observed.get("punishment", 0.0) or 0.0))
        observed_correctness = max(0.0, float(observed.get("correctness", 0.0) or 0.0))
        external_reward = max(0.0, float(external.get("reward", 0.0) or 0.0))
        external_punishment = max(0.0, float(external.get("punishment", 0.0) or 0.0))
        external_correctness = max(0.0, float(external.get("correctness", 0.0) or 0.0))
        observed_dominates_internal = bool(observed.get("dominates_internal_prediction", False))
        dominates_internal = bool(external.get("dominates_internal_prediction", False))
        observed_is_clear_punishment = bool(
            observed_dominates_internal
            and observed_punishment > (observed_reward + observed_correctness)
        )
        external_is_clear_punishment = bool(
            dominates_internal
            and external_punishment > (external_reward + external_correctness)
        )
        if external_is_clear_punishment:
            # A process teacher's post-action punishment should not be washed
            # out by AP's generic "prediction stabilized" self-reward. This is
            # still ordinary action feedback: it carries no answer unless the
            # caller explicitly supplied a post-action correction token.
            internal_positive_weight = 0.18
        else:
            internal_positive_weight = 1.0
        external_positive_weight = 0.18 if observed_is_clear_punishment else 1.0
        reward = observed_reward * internal_positive_weight + external_reward * external_positive_weight
        punishment = observed_punishment + external_punishment
        correctness = observed_correctness * internal_positive_weight + external_correctness * external_positive_weight
        confidence = max(float(observed.get("confidence", 0.0) or 0.0), float(external.get("confidence", 0.0) or 0.0))
        notes = list(observed.get("notes", []) or []) + [f"external_feedback::{external.get('source', 'external_feedback')}"]
        notes.extend([str(note or "") for note in list(external.get("notes", []) or []) if str(note or "")])
        if observed_is_clear_punishment or external_is_clear_punishment:
            notes.append("external_process_punishment_dominates_internal_prediction")
        merged = {
            "reward": round(reward, 4),
            "punishment": round(punishment, 4),
            "correctness": round(correctness, 4),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "notes": notes,
            "external_feedback": external,
        }
        def _explicit_feedback_value(key: str):
            for source in (external, observed):
                if key not in source:
                    continue
                value = source.get(key)
                if isinstance(value, bool):
                    if value:
                        return value
                    continue
                if value is not None and str(value) != "":
                    return value
            return None

        for key in (
            "feedback_expected_token",
            "teacher_reference_token_post_action_only",
            "target_token",
            "dominates_internal_prediction",
            "feedback_kind",
            "feedback_repair_intent",
        ):
            value = _explicit_feedback_value(key)
            if value is not None:
                merged[key] = value
        return merged

    def _build_explicit_feedback_items(self, feedback: dict) -> list[dict]:
        reward = max(0.0, float((feedback or {}).get("reward", 0.0) or 0.0))
        punishment = max(0.0, float((feedback or {}).get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float((feedback or {}).get("correctness", 0.0) or 0.0))
        confidence = max(0.0, min(1.0, float((feedback or {}).get("confidence", 0.0) or 0.0)))
        source = str((feedback or {}).get("source", "") or "external_feedback")
        items: list[dict] = []
        if reward > 0.0:
            items.append(
                {
                    "sa_label": "signal::reward",
                    "display_text": "外部奖励",
                    "source_type": "external_feedback",
                    "family": "signal",
                    "real_energy": round(reward, 4),
                    "anchor_meta": {
                        "schema_id": "explicit_feedback_signal/v1",
                        "feedback_kind": "reward",
                        "source": source,
                        "confidence": round(confidence, 4),
                        "observed_feedback": dict(feedback or {}),
                    },
                }
            )
        if correctness > 0.0:
            items.append(
                {
                    "sa_label": "signal::correctness",
                    "display_text": "外部正确性",
                    "source_type": "external_feedback",
                    "family": "signal",
                    "real_energy": round(correctness, 4),
                    "anchor_meta": {
                        "schema_id": "explicit_feedback_signal/v1",
                        "feedback_kind": "correctness",
                        "source": source,
                        "confidence": round(confidence, 4),
                        "observed_feedback": dict(feedback or {}),
                    },
                }
            )
        if punishment > 0.0:
            items.append(
                {
                    "sa_label": "signal::punishment",
                    "display_text": "外部惩罚",
                    "source_type": "external_feedback",
                    "family": "signal",
                    "real_energy": round(punishment, 4),
                    "virtual_energy": round(punishment, 4),
                    "anchor_meta": {
                        "schema_id": "explicit_feedback_signal/v1",
                        "feedback_kind": "punishment",
                        "source": source,
                        "confidence": round(confidence, 4),
                        "observed_feedback": dict(feedback or {}),
                        "feedback_energy_semantics": {
                            "real_energy": round(punishment, 4),
                            "virtual_energy": round(punishment, 4),
                            "punishment_pressure": round(punishment, 4),
                            "meaning": "punishment_event_as_real;future_avoidance_pressure_as_virtual",
                        },
                    },
                }
            )
        return items

    def _active_text_successor_cursor_ttl(self) -> int:
        visible_buffer = int(getattr(self.text_actuator, "max_visible_buffer", 12) or 12)
        return max(2, min(6, visible_buffer // 3))

    def _clear_active_text_successor_cursor(self, *, reason: str) -> dict:
        previous = dict(self._active_text_successor_cursor or {})
        self._active_text_successor_cursor = None
        return {
            "schema_id": "active_text_successor_cursor_clear/v1",
            "cleared": bool(previous),
            "reason": str(reason or ""),
            "previous": self._compact_text_successor_cursor(previous),
        }

    def _compact_text_successor_cursor(self, cursor: dict | None) -> dict:
        row = dict(cursor or {})
        if not row:
            return {}
        return {
            "schema_id": "active_text_successor_cursor/v1",
            "source_memory_id": str(row.get("source_memory_id", "") or ""),
            "successor_memory_id": str(row.get("successor_memory_id", "") or ""),
            "successor_memory_kind": str(row.get("successor_memory_kind", "") or ""),
            "successor_edge_kind": str(row.get("successor_edge_kind", "") or ""),
            "token": str(row.get("token", "") or ""),
            "visible_text_after": str(row.get("visible_text_after", "") or ""),
            "created_tick": int(row.get("created_tick", -1) or -1),
            "expires_at_tick": int(row.get("expires_at_tick", -1) or -1),
            "source_channel": str(row.get("source_channel", "") or ""),
            "branch_score": _round4(float(row.get("branch_score", 0.0) or 0.0)),
            "item_virtual_energy": _round4(float(row.get("item_virtual_energy", 0.0) or 0.0)),
        }

    def _read_active_text_successor_branch(self) -> dict:
        cursor = dict(self._active_text_successor_cursor or {})
        if not cursor:
            return {
                "schema_id": "active_text_successor_branch/v1",
                "available": False,
                "reason": "no_active_cursor",
                "rows": [],
            }
        expires_at = int(cursor.get("expires_at_tick", -1) or -1)
        if expires_at >= 0 and int(self.tick_index) > expires_at:
            cleared = self._clear_active_text_successor_cursor(reason="cursor_ttl_expired")
            return {
                "schema_id": "active_text_successor_branch/v1",
                "available": False,
                "reason": "cursor_ttl_expired",
                "clear": cleared,
                "rows": [],
            }
        source_id = str(cursor.get("successor_memory_id", "") or "")
        if not source_id:
            cleared = self._clear_active_text_successor_cursor(reason="cursor_missing_successor_memory_id")
            return {
                "schema_id": "active_text_successor_branch/v1",
                "available": False,
                "reason": "cursor_missing_successor_memory_id",
                "clear": cleared,
                "rows": [],
            }
        source_snapshot = self.memory.snapshot_by_id(source_id) or {}
        memory_kind = str(
            (source_snapshot or {}).get("memory_kind", "")
            or cursor.get("successor_memory_kind", "")
            or "focus"
        )
        rows = self.memory.successors(
            source_id,
            memory_kind=memory_kind,
            top_k=self.config.memory.predict_top_k,
            current_tick=None,
            source_b_row=dict(cursor.get("source_b_row", {}) or {}) or None,
        )
        if not rows and memory_kind != "focus":
            rows = self.memory.successors(
                source_id,
                memory_kind="focus",
                top_k=self.config.memory.predict_top_k,
                current_tick=None,
                source_b_row=dict(cursor.get("source_b_row", {}) or {}) or None,
            )
            memory_kind = "focus" if rows else memory_kind
        tagged_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tagged = dict(row)
            tagged["active_text_successor_cursor"] = self._compact_text_successor_cursor(cursor)
            tagged["prediction_source"] = "active_text_successor_branch"
            tagged_rows.append(tagged)
        if not tagged_rows:
            cleared = self._clear_active_text_successor_cursor(reason="cursor_successor_branch_empty")
            return {
                "schema_id": "active_text_successor_branch/v1",
                "available": False,
                "reason": "cursor_successor_branch_empty",
                "cursor": self._compact_text_successor_cursor(cursor),
                "clear": cleared,
                "rows": [],
            }
        return {
            "schema_id": "active_text_successor_branch/v1",
            "available": True,
            "reason": "follow_previous_text_insert_successor",
            "cursor": self._compact_text_successor_cursor(cursor),
            "source_memory_id": source_id,
            "memory_kind": memory_kind,
            "rows": tagged_rows,
            "row_refs": [self._cn_ref(row) for row in tagged_rows[:4]],
        }

    def _merge_successor_branch_rows(self, priority_rows: list[dict], existing_rows: list[dict]) -> list[dict]:
        merged = []
        seen: set[tuple[str, str]] = set()
        for row in list(priority_rows or []) + list(existing_rows or []):
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("source_memory_id", "") or ""),
                str(row.get("successor_memory_id", "") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(row))
        return merged

    def _update_active_text_successor_cursor(self, *, text_output_trace: dict, fast_cn: list[dict], slow_cn: list[dict]) -> dict:
        events = [dict(event) for event in list((text_output_trace or {}).get("recent_events", []) or []) if isinstance(event, dict)]
        if not events:
            return {
                "schema_id": "active_text_successor_cursor_update/v1",
                "updated": False,
                "reason": "no_text_event",
                "cursor": self._compact_text_successor_cursor(self._active_text_successor_cursor),
            }
        for event in reversed(events):
            event_type = str(event.get("event_type", "") or "")
            if event_type == "commit":
                cleared = self._clear_active_text_successor_cursor(reason="text_commit_closed_branch")
                return {
                    "schema_id": "active_text_successor_cursor_update/v1",
                    "updated": False,
                    "reason": "text_commit_closed_branch",
                    "clear": cleared,
                }
            if event_type in {"delete", "replace", "replace_noop"}:
                cleared = self._clear_active_text_successor_cursor(reason=f"text_{event_type}_changed_visible_prefix")
                return {
                    "schema_id": "active_text_successor_cursor_update/v1",
                    "updated": False,
                    "reason": f"text_{event_type}_changed_visible_prefix",
                    "clear": cleared,
                }
            if event_type in {"insert", "write_revision"}:
                branch = self._find_text_insert_successor_source(event=event, fast_cn=fast_cn, slow_cn=slow_cn)
                if not branch:
                    cleared = self._clear_active_text_successor_cursor(reason="insert_token_not_bound_to_cn_branch")
                    return {
                        "schema_id": "active_text_successor_cursor_update/v1",
                        "updated": False,
                        "reason": "insert_token_not_bound_to_cn_branch",
                        "event": {
                            "token": str(event.get("token", "") or ""),
                            "visible_text_before": str(event.get("visible_text_before", "") or ""),
                            "visible_text_after": str(event.get("visible_text_after", "") or ""),
                        },
                        "clear": cleared,
                    }
                successor_snapshot = self.memory.snapshot_by_id(str(branch.get("successor_memory_id", "") or "")) or {}
                ttl = self._active_text_successor_cursor_ttl()
                cursor = {
                    "schema_id": "active_text_successor_cursor/v1",
                    "source_memory_id": str(branch.get("source_memory_id", "") or ""),
                    "successor_memory_id": str(branch.get("successor_memory_id", "") or ""),
                    "successor_memory_kind": str((successor_snapshot or {}).get("memory_kind", "") or "focus"),
                    "successor_edge_kind": str(branch.get("successor_edge_kind", "") or ""),
                    "token": str(event.get("token", "") or ""),
                    "visible_text_before": str(event.get("visible_text_before", "") or ""),
                    "visible_text_after": str(event.get("visible_text_after", "") or ""),
                    "created_tick": int(self.tick_index),
                    "expires_at_tick": int(self.tick_index) + ttl,
                    "source_channel": str(branch.get("source_channel", "") or ""),
                    "branch_score": float(branch.get("branch_score", 0.0) or 0.0),
                    "item_virtual_energy": float(branch.get("item_virtual_energy", 0.0) or 0.0),
                    "source_b_row": dict(branch.get("source_b_row", {}) or {}),
                    "reason": "selected_text_insert_followed_cn_successor",
                    "policy": "follow_explicit_successor_branch_after_actual_text_insert",
                }
                self._active_text_successor_cursor = cursor
                return {
                    "schema_id": "active_text_successor_cursor_update/v1",
                    "updated": True,
                    "reason": "insert_bound_to_cn_branch",
                    "cursor": self._compact_text_successor_cursor(cursor),
                    "event": {
                        "token": str(event.get("token", "") or ""),
                        "visible_text_after": str(event.get("visible_text_after", "") or ""),
                    },
                }
            if event_type == "reread":
                branch = self._find_text_action_successor_source(event=event, fast_cn=fast_cn, slow_cn=slow_cn)
                if branch:
                    successor_snapshot = self.memory.snapshot_by_id(str(branch.get("successor_memory_id", "") or "")) or {}
                    ttl = self._active_text_successor_cursor_ttl()
                    cursor = {
                        "schema_id": "active_text_successor_cursor/v1",
                        "source_memory_id": str(branch.get("source_memory_id", "") or ""),
                        "successor_memory_id": str(branch.get("successor_memory_id", "") or ""),
                        "successor_memory_kind": str((successor_snapshot or {}).get("memory_kind", "") or "focus"),
                        "successor_edge_kind": str(branch.get("successor_edge_kind", "") or ""),
                        "token": str(event.get("token", "") or ""),
                        "visible_text_before": str(event.get("visible_text_before", "") or ""),
                        "visible_text_after": str(event.get("visible_text_after", "") or ""),
                        "created_tick": int(self.tick_index),
                        "expires_at_tick": int(self.tick_index) + ttl,
                        "source_channel": str(branch.get("source_channel", "") or ""),
                        "branch_score": float(branch.get("branch_score", 0.0) or 0.0),
                        "item_virtual_energy": float(branch.get("item_virtual_energy", 0.0) or 0.0),
                        "source_b_row": dict(branch.get("source_b_row", {}) or {}),
                        "reason": "selected_text_reread_followed_cn_successor",
                        "policy": "follow_explicit_successor_branch_after_actual_text_reread",
                    }
                    self._active_text_successor_cursor = cursor
                    return {
                        "schema_id": "active_text_successor_cursor_update/v1",
                        "updated": True,
                        "reason": "reread_bound_to_cn_branch",
                        "cursor": self._compact_text_successor_cursor(cursor),
                        "event": {
                            "token": str(event.get("token", "") or ""),
                            "visible_text_after": str(event.get("visible_text_after", "") or ""),
                        },
                    }
        return {
            "schema_id": "active_text_successor_cursor_update/v1",
            "updated": False,
            "reason": "no_branch_changing_text_event",
            "cursor": self._compact_text_successor_cursor(self._active_text_successor_cursor),
        }

    def _find_text_insert_successor_source(self, *, event: dict, fast_cn: list[dict], slow_cn: list[dict]) -> dict:
        token = str((event or {}).get("token", "") or "")
        if not token:
            return {}
        candidates = []
        for source_rank, (channel, branches) in enumerate((("slow_cn", slow_cn), ("fast_cn", fast_cn))):
            for branch_index, branch in enumerate(list(branches or [])):
                if not isinstance(branch, dict):
                    continue
                successor_id = str(branch.get("successor_memory_id", "") or "")
                if not successor_id:
                    continue
                for item_index, item in enumerate(list(branch.get("predicted_items", []) or [])):
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("sa_label", "") or "")
                    if label != f"text::{token}":
                        continue
                    alignment = self._text_prediction_alignment_for_event(item=item, event=event)
                    if alignment == "misaligned":
                        continue
                    try:
                        item_energy = float(item.get("virtual_energy", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        item_energy = 0.0
                    try:
                        branch_score = float(branch.get("score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        branch_score = 0.0
                    try:
                        source_b_weight = float(branch.get("source_b_weight", branch.get("successor_normalized_weight", 0.0)) or 0.0)
                    except (TypeError, ValueError):
                        source_b_weight = 0.0
                    try:
                        source_b_efficiency = float(branch.get("source_b_match_efficiency", branch.get("successor_normalized_weight", 0.0)) or 0.0)
                    except (TypeError, ValueError):
                        source_b_efficiency = 0.0
                    alignment_rank = 0 if alignment == "aligned" else 1
                    candidates.append(
                        {
                            "source_memory_id": str(branch.get("source_memory_id", "") or ""),
                            "successor_memory_id": successor_id,
                            "successor_edge_kind": str(branch.get("successor_edge_kind", "") or ""),
                            "source_channel": channel,
                            "branch_score": branch_score,
                            "source_b_weight": source_b_weight,
                            "source_b_match_efficiency": source_b_efficiency,
                            "item_virtual_energy": item_energy,
                            "alignment": alignment,
                            "source_b_row": self._active_text_successor_source_b_row(
                                branch=branch,
                                successor_id=successor_id,
                                item_energy=item_energy,
                            ),
                            "sort_key": (
                                alignment_rank,
                                -branch_score,
                                -source_b_weight,
                                -source_b_efficiency,
                                source_rank,
                                -item_energy,
                                branch_index,
                                item_index,
                            ),
                        }
                    )
        if not candidates:
            return {}
        candidates.sort(key=lambda row: row["sort_key"])
        best = dict(candidates[0])
        best.pop("sort_key", None)
        return best

    def _find_text_action_successor_source(self, *, event: dict, fast_cn: list[dict], slow_cn: list[dict]) -> dict:
        event_type = str((event or {}).get("event_type", "") or "")
        action_id = str((event or {}).get("action_id", "") or "")
        token = str((event or {}).get("token", "") or "")
        if not event_type or not action_id:
            return {}
        candidates = []
        for source_rank, (channel, branches) in enumerate((("slow_cn", slow_cn), ("fast_cn", fast_cn))):
            for branch_index, branch in enumerate(list(branches or [])):
                if not isinstance(branch, dict):
                    continue
                successor_id = str(branch.get("successor_memory_id", "") or "")
                if not successor_id:
                    continue
                for item_index, item in enumerate(list(branch.get("predicted_items", []) or [])):
                    if not isinstance(item, dict):
                        continue
                    if not self._text_action_prediction_matches_event(item=item, event=event):
                        continue
                    try:
                        item_energy = float(item.get("virtual_energy", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        item_energy = 0.0
                    try:
                        branch_score = float(branch.get("score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        branch_score = 0.0
                    try:
                        source_b_weight = float(branch.get("source_b_weight", branch.get("successor_normalized_weight", 0.0)) or 0.0)
                    except (TypeError, ValueError):
                        source_b_weight = 0.0
                    try:
                        source_b_efficiency = float(branch.get("source_b_match_efficiency", branch.get("successor_normalized_weight", 0.0)) or 0.0)
                    except (TypeError, ValueError):
                        source_b_efficiency = 0.0
                    candidates.append(
                        {
                            "source_memory_id": str(branch.get("source_memory_id", "") or ""),
                            "successor_memory_id": successor_id,
                            "successor_edge_kind": str(branch.get("successor_edge_kind", "") or ""),
                            "source_channel": channel,
                            "branch_score": branch_score,
                            "source_b_weight": source_b_weight,
                            "source_b_match_efficiency": source_b_efficiency,
                            "item_virtual_energy": item_energy,
                            "event_type": event_type,
                            "action_id": action_id,
                            "token": token,
                            "source_b_row": self._active_text_successor_source_b_row(
                                branch=branch,
                                successor_id=successor_id,
                                item_energy=item_energy,
                            ),
                            "sort_key": (
                                -branch_score,
                                -source_b_weight,
                                -source_b_efficiency,
                                source_rank,
                                -item_energy,
                                branch_index,
                                item_index,
                            ),
                        }
                    )
        if not candidates:
            return {}
        candidates.sort(key=lambda row: row["sort_key"])
        best = dict(candidates[0])
        best.pop("sort_key", None)
        return best

    def _text_action_prediction_matches_event(self, *, item: dict, event: dict) -> bool:
        label = str((item or {}).get("sa_label", "") or "")
        meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        event_type = str((event or {}).get("event_type", "") or "")
        action_id = str((event or {}).get("action_id", "") or "")
        token = str((event or {}).get("token", "") or "")
        if action_id and label == action_id:
            return True
        if event_type and label.startswith(f"text_action::{event_type}"):
            predicted_token = str(meta.get("token", "") or meta.get("candidate_token", "") or meta.get("expected_token", "") or "")
            if not predicted_token and label.startswith(f"text_action::{event_type}::"):
                predicted_token = label.split("::", 2)[-1]
            return bool(not token or not predicted_token or token == predicted_token)
        source_event_type = str(meta.get("source_event_type", "") or meta.get("event_type", "") or "")
        meta_action_id = str(meta.get("action_id", "") or "")
        return bool(source_event_type == event_type and meta_action_id == action_id)

    def _active_text_successor_source_b_row(self, *, branch: dict, successor_id: str, item_energy: float) -> dict:
        inherited_virtual = max(0.0, float(item_energy or 0.0))
        for key in ("source_b_effective_virtual_energy", "source_b_virtual_energy"):
            try:
                inherited_virtual = max(inherited_virtual, float((branch or {}).get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        try:
            source_b_weight = float(branch.get("source_b_weight", branch.get("successor_normalized_weight", 0.0)) or 0.0)
        except (TypeError, ValueError):
            source_b_weight = 0.0
        try:
            source_b_efficiency = float(branch.get("source_b_match_efficiency", branch.get("successor_normalized_weight", 0.0)) or 0.0)
        except (TypeError, ValueError):
            source_b_efficiency = 0.0
        return {
            "memory_id": str(successor_id or ""),
            "normalized_weight": _round4(max(0.0, source_b_weight)),
            "match_efficiency": _round4(max(0.0, source_b_efficiency)),
            "grasp_confidence": _round4(max(0.0, source_b_efficiency)),
            "b_real_energy": 0.0,
            "b_virtual_energy": _round4(inherited_virtual),
            "b_effective_real_energy": 0.0,
            "b_effective_virtual_energy": _round4(inherited_virtual),
            "energy_transfer": {
                "schema_id": "active_text_successor_energy_inheritance/v1",
                "source_memory_id": str((branch or {}).get("source_memory_id", "") or ""),
                "successor_memory_id": str(successor_id or ""),
                "inherited_virtual_energy": _round4(inherited_virtual),
                "policy": "actual_inserted_cn_energy_becomes_short_lived_branch_process_support",
            },
        }

    def _text_prediction_alignment_for_event(self, *, item: dict, event: dict) -> str:
        meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        has_position = any(key in meta for key in ("current_glyph_index", "visible_length", "cursor", "cursor_index", "previous_prefix"))
        if not has_position:
            return "fallback"
        visible_before = str((event or {}).get("visible_text_before", "") or "")
        visible_length = len(visible_before)
        for key in ("visible_length", "current_glyph_index", "cursor", "cursor_index"):
            if key not in meta:
                continue
            try:
                if int(meta.get(key)) != int(visible_length):
                    return "misaligned"
            except (TypeError, ValueError):
                return "misaligned"
        previous_prefix = str(meta.get("previous_prefix", "") or "")
        if previous_prefix and previous_prefix != visible_before:
            return "misaligned"
        token = str((event or {}).get("token", "") or "")
        variant_text = str(meta.get("variant_text", "") or meta.get("expected_text", "") or "")
        if variant_text:
            if not variant_text.startswith(visible_before):
                return "misaligned"
            if not variant_text[len(visible_before) :].startswith(token):
                return "misaligned"
        return "aligned"

    def _run_recall_branch(
        self,
        query_items: list[dict],
        *,
        memory_kind: str,
        prediction_source: str,
        time_context: dict | None = None,
    ) -> tuple[list[dict], list[dict]]:
        primary_bn = self.memory.recall(query_items, memory_kind=memory_kind, time_context=time_context)
        if self._should_deepen_b_recall(primary_bn, memory_kind=memory_kind):
            bn_rows = self.memory.recall_residual(query_items, memory_kind=memory_kind, time_context=time_context)
        else:
            bn_rows = primary_bn
        cn_rows = []
        for row in bn_rows:
            cn_rows.extend(
                self.memory.successors(
                    row["memory_id"],
                    memory_kind=memory_kind,
                    source_b_row=row,
                    current_tick=self.tick_index,
                )
            )
        predicted_items = [item for branch in cn_rows for item in branch.get("predicted_items", [])]
        if predicted_items:
            self.state_pool.apply_predictions(predicted_items, tick_index=self.tick_index, source=prediction_source)
        return bn_rows, cn_rows

    def _should_deepen_b_recall(self, primary_bn: list[dict], *, memory_kind: str) -> bool:
        if str(memory_kind or "") == "focus" and not bool(getattr(self.config.memory, "slow_residual_recall_enabled", True)):
            return False
        threshold = float(getattr(self.config.memory, "residual_recall_deepen_grasp_threshold", 1.01) or 1.01)
        min_rows = max(1, int(getattr(self.config.memory, "residual_recall_min_primary_rows", 1) or 1))
        rows = [dict(row) for row in list(primary_bn or []) if isinstance(row, dict)]
        if len(rows) < min_rows:
            return True
        best_grasp = 0.0
        for row in rows:
            best_grasp = max(
                best_grasp,
                float(row.get("grasp_confidence", row.get("match_efficiency", 0.0)) or 0.0),
            )
        return best_grasp < threshold

    def _build_slow_query(self, selected_focus_items: list[dict], *, action_slow_query_hints: list[dict] | None = None) -> list[dict]:
        focus_labels = []
        seen_focus = set()
        for item in selected_focus_items or []:
            label = str((item or {}).get("sa_label", "") or "")
            if label and label not in seen_focus:
                seen_focus.add(label)
                focus_labels.append(label)
        for label in self.focus_buffer.recent_labels():
            if label and label not in seen_focus:
                seen_focus.add(label)
                focus_labels.append(label)
        action_hint_labels = [
            str(hint.get("sa_label", "") or "")
            for hint in list(action_slow_query_hints or [])
            if isinstance(hint, dict) and str(hint.get("sa_label", "") or "")
        ]
        for label in action_hint_labels:
            if label and label not in seen_focus:
                seen_focus.add(label)
                focus_labels.append(label)
        state_rows = self.state_pool.rows_for_labels(focus_labels)
        slot_packet_items = self._current_short_term_slot_items()
        if slot_packet_items:
            state_rows = self._merge_state_rows(state_rows, slot_packet_items)
        state_by_label = {str(item.get("sa_label", "") or ""): item for item in state_rows if str(item.get("sa_label", "") or "")}
        query_rows = []
        current_order: dict[str, int] = {}
        selected_labels = []
        seen_selected = set()
        for item in selected_focus_items or []:
            label = str((item or {}).get("sa_label", "") or "")
            if not label or label in seen_selected:
                continue
            seen_selected.add(label)
            current_order[label] = len(selected_labels)
            selected_labels.append(label)
        for label in selected_labels:
            state_row = state_by_label.get(label)
            if state_row is not None:
                current_row = dict(state_row)
                current_row["source_type"] = str(current_row.get("source_type", "") or "current_focus")
                current_row["query_source"] = "current_focus"
                current_row["query_source_priority"] = 1
                current_row["focus_order_index"] = int(current_order.get(label, len(current_order)))
                if "query_weight" not in current_row:
                    current_row["query_weight"] = float(current_row.get("real_energy", 0.0) or 0.0)
                query_rows.append(current_row)
        continuation_rows = self.focus_buffer.build_query_items(state_rows, tick_index=self.tick_index)
        replay_rows = self.focus_buffer.build_replay_query_items(state_rows, tick_index=self.tick_index)
        merged: dict[str, dict] = {}
        action_hint_rows = []
        hint_by_label = {
            str(hint.get("sa_label", "") or ""): dict(hint)
            for hint in list(action_slow_query_hints or [])
            if isinstance(hint, dict) and str(hint.get("sa_label", "") or "")
        }
        for label, hint in hint_by_label.items():
            state_row = state_by_label.get(label)
            if state_row is None:
                continue
            row = dict(state_row)
            row["source_type"] = "action_control_hint"
            row["query_source"] = "action_control_hint"
            if str(hint.get("control_kind", "") or "") == "draft_surface_continuation":
                row["query_source_priority"] = 0
            else:
                row["query_source_priority"] = 2
            row["query_weight"] = max(float(row.get("query_weight", 0.0) or 0.0), float(hint.get("query_weight", 0.0) or 0.0))
            row["virtual_energy"] = max(float(row.get("virtual_energy", 0.0) or 0.0), float(hint.get("virtual_energy", 0.0) or 0.0))
            row["anchor_meta"] = {"action_slow_query_hint": hint}
            action_hint_rows.append(row)
        for row in query_rows + continuation_rows + replay_rows + action_hint_rows:
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            bucket = merged.get(label)
            if bucket is None:
                item = dict(row)
                item["query_sources"] = [str(item.get("query_source", item.get("source_type", "")) or "unknown")]
                if label in current_order:
                    item["focus_order_index"] = int(current_order[label])
                item["query_source_priority"] = int(item.get("query_source_priority", 2 if label not in current_order else 1) or 0)
                merged[label] = item
                continue
            source_name = str(row.get("query_source", row.get("source_type", "")) or "unknown")
            sources = list(bucket.get("query_sources", []) or [])
            if source_name not in sources:
                sources.append(source_name)
            bucket["query_sources"] = sources
            if source_name == "focus_replay" and "current_focus" not in sources:
                bucket["source_type"] = "focus_replay"
            bucket["query_source_priority"] = min(
                int(bucket.get("query_source_priority", 2 if label not in current_order else 1) or 0),
                int(row.get("query_source_priority", 2 if label not in current_order else 1) or 0),
            )
            bucket["query_weight"] = max(float(bucket.get("query_weight", 0.0) or 0.0), float(row.get("query_weight", 0.0) or 0.0))
            bucket["real_energy"] = max(float(bucket.get("real_energy", 0.0) or 0.0), float(row.get("real_energy", 0.0) or 0.0))
            bucket["virtual_energy"] = max(float(bucket.get("virtual_energy", 0.0) or 0.0), float(row.get("virtual_energy", 0.0) or 0.0))
            bucket["cognitive_pressure"] = float(bucket.get("real_energy", 0.0) or 0.0) - float(bucket.get("virtual_energy", 0.0) or 0.0)
        rows = list(merged.values())
        rows.sort(
            key=lambda item: (
                int(item.get("query_source_priority", 2 if str(item.get("sa_label", "") or "") not in current_order else 1) or 0),
                0 if str(item.get("sa_label", "") or "") in current_order else 1,
                int(current_order.get(str(item.get("sa_label", "") or ""), 10**9)),
                -float(item.get("query_weight", item.get("real_energy", 0.0)) or 0.0),
                str(item.get("sa_label", "") or ""),
            )
        )
        return rows

    def _current_short_term_slot_items(self) -> list[dict]:
        trace = getattr(self, "_last_short_term_slot_trace", None)
        if isinstance(trace, dict):
            items = list(trace.get("items", []) or [])
            return [dict(item) for item in items if isinstance(item, dict)]
        return []

    def _merge_state_rows(self, primary_rows: list[dict], extra_rows: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for row in list(primary_rows or []) + list(extra_rows or []):
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            existing = merged.get(label)
            if existing is None:
                merged[label] = dict(row)
                continue
            existing["real_energy"] = max(float(existing.get("real_energy", 0.0) or 0.0), float(row.get("real_energy", 0.0) or 0.0))
            existing["virtual_energy"] = max(float(existing.get("virtual_energy", 0.0) or 0.0), float(row.get("virtual_energy", 0.0) or 0.0))
            existing["attention_gain"] = max(float(existing.get("attention_gain", 0.0) or 0.0), float(row.get("attention_gain", 0.0) or 0.0))
            if isinstance(row.get("anchor_meta"), dict):
                existing_meta = dict(existing.get("anchor_meta", {}) or {}) if isinstance(existing.get("anchor_meta", {}), dict) else {}
                existing_meta.update(dict(row.get("anchor_meta", {}) or {}))
                existing["anchor_meta"] = existing_meta
        rows = list(merged.values())
        rows.sort(key=lambda item: (-float(item.get("virtual_energy", 0.0) or 0.0), -float(item.get("real_energy", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        return rows

    def _build_action_control_items(
        self,
        *,
        selected_actions: list[dict],
        attention_trace: dict,
        fast_bn: list[dict],
        slow_bn: list[dict],
        fast_cn: list[dict],
        slow_cn: list[dict],
        state_snapshot_items: list[dict] | None = None,
        time_context: dict | None = None,
        action_consequence_trace: dict | None = None,
        expectation_pressure_trace: dict | None = None,
        focus_continuation_trace: dict | None = None,
        short_term_memory_recall: dict | None = None,
    ) -> list[dict]:
        items = []
        if not selected_actions:
            return items
        focus_labels = [label for label in (attention_trace.get("selected_labels", []) or []) if self._is_core_trace_label(str(label or ""))]
        replay_labels = self._extract_replay_labels(fast_bn=fast_bn, slow_bn=slow_bn, focus_labels=focus_labels)
        stabilize_labels = self._extract_predicted_labels(fast_cn=fast_cn, slow_cn=slow_cn)
        for row in selected_actions:
            action_id = str(row.get("action_id", "") or "")
            if action_id == "action::continue_focus":
                for label in focus_labels[:3]:
                    items.append(
                        {
                            "sa_label": label,
                            "display_text": label,
                            "family": "action_control",
                            "source_type": "action_control",
                            "virtual_energy": 0.42,
                            "anchor_meta": {"action_id": action_id, "control_kind": "continue_focus"},
                        }
                    )
            elif action_id in {"action::replay_recent_context", "action::recall_recent_context"}:
                if action_id == "action::recall_recent_context":
                    items.extend(
                        self._short_term_memory_control_rows(
                            selected_action=row,
                            recall_trace=short_term_memory_recall or {},
                            limit=8,
                        )
                    )
                    items.extend(
                        self._recent_thought_readback_control_rows(
                            selected_action=row,
                            focus_continuation_trace=focus_continuation_trace or {},
                            fallback_labels=replay_labels,
                            limit=8,
                        )
                    )
                    continue
                for label in replay_labels[:4]:
                    items.append(
                        {
                            "sa_label": label,
                            "display_text": label,
                            "family": "action_control",
                            "source_type": "action_control",
                            "virtual_energy": 0.58,
                            "anchor_meta": {"action_id": action_id, "control_kind": "replay_recent_context"},
                        }
                    )
                # The legacy replay action often wins the memory-recall lane in
                # real traces. When a short-term readback view is available, we
                # keep the replay rows above and also expose the self-observation
                # part: AP is effectively checking what it was just thinking.
                items.extend(
                    self._short_term_memory_control_rows(
                        selected_action=row,
                        recall_trace=short_term_memory_recall or {},
                        limit=8,
                    )
                )
                items.extend(
                    self._recent_thought_readback_control_rows(
                        selected_action=row,
                        focus_continuation_trace=focus_continuation_trace or {},
                        fallback_labels=replay_labels,
                        limit=8,
                    )
                )
            elif action_id == "action::recall_by_expectation":
                for control in self._expectation_recall_control_rows(
                    selected_action=row,
                    expectation_pressure_trace=expectation_pressure_trace or {},
                    limit=6,
                ):
                    items.append(control)
            elif action_id == "action::recall_by_timefelt":
                items.extend(
                    self._timefelt_recall_control_rows(
                        selected_action=row,
                        state_snapshot_items=state_snapshot_items or [],
                        time_context=time_context,
                        limit=8,
                    )
                )
            elif action_id == "action::replay_episode":
                items.extend(
                    self._episode_replay_control_rows(
                        selected_action=row,
                        expectation_pressure_trace=expectation_pressure_trace or {},
                        action_consequence_trace=action_consequence_trace or {},
                        limit=10,
                    )
                )
            elif action_id == "action::wait":
                items.append(self._wait_control_row(selected_action=row))
            elif action_id == "action::stabilize_prediction":
                for label in stabilize_labels[:4]:
                    items.append(
                        {
                            "sa_label": label,
                            "display_text": label,
                            "family": "action_control",
                            "source_type": "action_control",
                            "virtual_energy": 0.46,
                            "anchor_meta": {"action_id": action_id, "control_kind": "stabilize_prediction"},
                        }
                    )
        return items

    def _recent_thought_readback_control_rows(
        self,
        *,
        selected_action: dict,
        focus_continuation_trace: dict,
        fallback_labels: list[str],
        limit: int,
    ) -> list[dict]:
        readback = dict((focus_continuation_trace or {}).get("recent_thought_readback", {}) or {})
        params = dict((selected_action or {}).get("params", {}) or {})
        labels = [str(label or "") for label in list(readback.get("labels", []) or params.get("labels", []) or fallback_labels or []) if str(label or "")]
        seen = set()
        unique_labels = []
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            unique_labels.append(label)
            if len(unique_labels) >= max(1, int(limit)):
                break
        if not unique_labels:
            return []
        strength = self._selected_action_strength(selected_action)
        meta = {
            "schema_id": "recent_thought_readback_control/v1",
            "action_id": "action::recall_recent_context",
            "control_kind": "recent_thought_readback",
            "source_action_id": str((selected_action or {}).get("action_id", "") or "action::recall_recent_context"),
            "recalled_labels": unique_labels,
            "entries": list(readback.get("entries", []) or [])[:6],
            "active_episode_id": int(readback.get("active_episode_id", params.get("active_episode_id", -1)) or -1),
            "drift_score": round(float(readback.get("drift_score", 0.0) or 0.0), 4),
            "branch_end_score": round(float(readback.get("branch_end_score", 0.0) or 0.0), 4),
            "strength": round(strength, 4),
            "learning_boundary": "short_term_readback_modulates_attention_and_slow_query_not_forced_answer",
            "meaning": "AP_reading_its_recent_focus_episode_like_checking_what_it_was_thinking",
        }
        rows: list[dict] = [
            {
                "sa_label": "control::recent_thought_readback",
                "display_text": "recent thought readback",
                "family": "action_control",
                "source_type": "action_control",
                "real_energy": 0.0,
                "virtual_energy": round(min(0.68, 0.18 + strength * 0.42), 4),
                "anchor_meta": meta,
            }
        ]
        for label in unique_labels:
            rows.append(
                {
                    "sa_label": label,
                    "display_text": label,
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": round(min(0.52, 0.10 + strength * 0.30), 4),
                    "anchor_meta": {**meta, "target_label": label, "target_modulation": "recent_thought_readback"},
                }
            )
        return rows

    def _short_term_memory_control_rows(self, *, selected_action: dict, recall_trace: dict, limit: int) -> list[dict]:
        recall = dict(recall_trace or {})
        selected_items = [dict(item) for item in list(recall.get("selected_items", []) or []) if isinstance(item, dict)]
        if not selected_items:
            return []
        strength = self._selected_action_strength(selected_action)
        labels = []
        seen = set()
        for item in selected_items:
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if len(labels) >= max(1, int(limit)):
                break
        if not labels:
            return []
        meta = {
            "schema_id": "short_term_memory_recall_control/v1",
            "action_id": "action::recall_recent_context",
            "source_action_id": str((selected_action or {}).get("action_id", "") or "action::recall_recent_context"),
            "control_kind": "short_term_memory_recall",
            "recalled_labels": labels,
            "selected_events": [
                {
                    "event_id": str(event.get("event_id", "") or ""),
                    "tick_index": int(event.get("tick_index", -1) or -1),
                    "source_kind": str(event.get("source_kind", "") or ""),
                    "modality": str(event.get("modality", "") or ""),
                    "score": round(float(event.get("score", 0.0) or 0.0), 4),
                }
                for event in list(recall.get("selected_events", []) or [])[:4]
                if isinstance(event, dict)
            ],
            "cue_tokens": list(recall.get("cue_tokens", []) or [])[:12],
            "strength": round(strength, 4),
            "not_new_external_input": True,
            "learning_boundary": "short_term_memory_readback_modulates_attention_and_slow_query_not_forced_answer",
            "meaning": "AP_actively_recalls_a_recent_multimodal_working_memory_segment",
        }
        rows: list[dict] = [
            {
                "sa_label": "control::short_term_memory_recall",
                "display_text": "short-term memory recall",
                "family": "action_control",
                "source_type": "action_control",
                "real_energy": 0.0,
                "virtual_energy": round(min(0.72, 0.20 + strength * 0.42), 4),
                "anchor_meta": meta,
            }
        ]
        for item in selected_items[: max(1, int(limit))]:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            item_strength = max(0.05, min(1.0, float(item.get("recall_strength", strength) or strength)))
            rows.append(
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": round(min(0.56, 0.08 + item_strength * 0.34), 4),
                    "anchor_meta": {
                        **meta,
                        "target_label": label,
                        "target_modulation": "short_term_memory_recall",
                        "origin_tick_index": int(item.get("origin_tick_index", -1) or -1),
                        "source_kind": str(item.get("source_kind", "") or ""),
                        "modality": str(item.get("modality", "") or ""),
                        "event_id": str(item.get("event_id", "") or ""),
                    },
                }
            )
        return rows

    def _expectation_recall_control_rows(self, *, selected_action: dict, expectation_pressure_trace: dict, limit: int) -> list[dict]:
        anchors = [dict(anchor) for anchor in list((selected_action or {}).get("supporting_anchors", []) or []) if isinstance(anchor, dict)]
        if not anchors:
            trace = dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {})
            wanted = str(((selected_action or {}).get("params", {}) or {}).get("b_anchor", "") or "")
            anchors = [
                dict(anchor)
                for anchor in list(trace.get("anchors", []) or [])
                if isinstance(anchor, dict) and (not wanted or str(anchor.get("anchor_id", "") or "") == wanted)
            ]
        anchors.sort(
            key=lambda anchor: (
                -float(anchor.get("level", 0.0) or 0.0),
                0 if str(anchor.get("anchor_type", "") or "") == "pressure" else 1,
                str(anchor.get("anchor_id", "") or ""),
            )
        )
        rows: list[dict] = []
        seen: set[str] = set()
        for anchor in anchors:
            if len(rows) >= max(1, int(limit)):
                break
            source_memory_id = str(anchor.get("source_memory_id", "") or "")
            if not source_memory_id:
                continue
            snapshot = self.memory.snapshot_by_id(source_memory_id) or {}
            labels = self._labels_from_memory_snapshot(snapshot)
            for label in labels:
                if len(rows) >= max(1, int(limit)):
                    break
                if label in seen:
                    continue
                seen.add(label)
                level = max(0.08, min(1.0, float(anchor.get("level", 0.0) or 0.0)))
                rows.append(
                    {
                        "sa_label": label,
                        "display_text": label,
                        "family": "action_control",
                        "source_type": "action_control",
                        "virtual_energy": round(min(0.72, 0.24 + level * 0.46), 4),
                        "anchor_meta": {
                            "schema_id": "expectation_recall_control/v1",
                            "action_id": "action::recall_by_expectation",
                            "control_kind": "recall_by_expectation",
                            "anchor_id": str(anchor.get("anchor_id", "") or ""),
                            "anchor_type": str(anchor.get("anchor_type", "") or ""),
                            "source_memory_id": source_memory_id,
                            "source_memory_kind": str(anchor.get("source_memory_kind", "") or ""),
                            "source_tick_index": int(anchor.get("source_tick_index", -1) or -1),
                            "anchor_level": round(level, 4),
                            "expected_reward": round(float(anchor.get("expected_reward", 0.0) or 0.0), 4),
                            "expected_punishment": round(float(anchor.get("expected_punishment", 0.0) or 0.0), 4),
                            "recalled_from_snapshot": bool(snapshot),
                        },
                    }
                    )
        return rows

    def _timefelt_recall_control_rows(
        self,
        *,
        selected_action: dict,
        state_snapshot_items: list[dict],
        time_context: dict | None,
        limit: int,
    ) -> list[dict]:
        if not time_context:
            return []
        query_items = [
            dict(item)
            for item in list(state_snapshot_items or [])[: max(1, self.config.memory.query_feature_limit)]
            if isinstance(item, dict) and self._is_core_trace_label(str(item.get("sa_label", "") or ""))
        ]
        if not query_items:
            params = dict((selected_action or {}).get("params", {}) or {})
            query_items = [
                {"sa_label": label, "display_text": label, "family": "text", "source_type": "timefelt_query", "real_energy": 0.1}
                for label in list(params.get("query_labels", []) or [])[:4]
            ]
        state_rows = self.memory.recall(query_items, memory_kind="state", top_k=4, time_context=time_context) if query_items else []
        focus_rows = self.memory.recall(query_items, memory_kind="focus", top_k=3, time_context=time_context) if query_items else []
        recall_rows = sorted(
            [dict(row) for row in list(state_rows or []) + list(focus_rows or [])],
            key=lambda item: (-float(item.get("time_match", 0.0) or 0.0), -float(item.get("score", 0.0) or 0.0), str(item.get("memory_id", "") or "")),
        )
        if not recall_rows:
            return []
        strength = self._selected_action_strength(selected_action)
        target_delta_t = float(time_context.get("target_delta_t", 0.0) or 0.0)
        sigma = float(time_context.get("time_sigma", 1.0) or 1.0)
        source_memory_ids = []
        labels: list[str] = []
        seen_labels: set[str] = set()
        for row in recall_rows[:4]:
            memory_id = str(row.get("memory_id", "") or "")
            if memory_id and memory_id not in source_memory_ids:
                source_memory_ids.append(memory_id)
            snapshot = dict(row.get("snapshot", {}) or self.memory.snapshot_by_id(memory_id) or {})
            for label in self._labels_from_memory_snapshot(snapshot, limit=max(2, int(limit))):
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                labels.append(label)
                if len(labels) >= max(1, int(limit)):
                    break
            if len(labels) >= max(1, int(limit)):
                break
        meta = {
            "schema_id": "timefelt_recall_control/v1",
            "action_id": "action::recall_by_timefelt",
            "control_kind": "recall_by_timefelt",
            "source_action_id": "action::recall_by_timefelt",
            "target_delta_t": round(target_delta_t, 4),
            "time_sigma": round(max(1.0, sigma), 4),
            "source_memory_ids": source_memory_ids,
            "replayed_labels": labels,
            "strength": round(strength, 4),
            "learning_boundary": "timefelt_recall_modulates_attention_and_slow_query_not_concept_embedding",
            "humanlike_testing": {
                "engineering_latency_ticks": "1-2",
                "behavior_window_ticks": "5-10",
            },
        }
        rows: list[dict] = [
            {
                "sa_label": "control::timefelt_recall",
                "display_text": "时间感回忆",
                "family": "action_control",
                "source_type": "action_control",
                "real_energy": 0.0,
                "virtual_energy": round(min(0.72, 0.18 + strength * 0.46), 4),
                "anchor_meta": meta,
            }
        ]
        for label in labels:
            rows.append(
                {
                    "sa_label": label,
                    "display_text": label,
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": round(min(0.58, 0.12 + strength * 0.36), 4),
                    "anchor_meta": {**meta, "target_label": label, "target_modulation": "timefelt_recall"},
                }
            )
        return rows

    def _episode_replay_control_rows(
        self,
        *,
        selected_action: dict,
        expectation_pressure_trace: dict,
        action_consequence_trace: dict,
        limit: int,
    ) -> list[dict]:
        source = self._select_episode_replay_source(
            selected_action=selected_action,
            expectation_pressure_trace=expectation_pressure_trace,
            action_consequence_trace=action_consequence_trace,
        )
        source_memory_id = str(source.get("source_memory_id", "") or "")
        snapshot = self.memory.snapshot_by_id(source_memory_id) if source_memory_id else None
        if not snapshot:
            return []
        params = dict((selected_action or {}).get("params", {}) or {})
        episode_id = params.get("episode_id", params.get("source_episode_id", None))
        if episode_id is not None:
            try:
                self.focus_buffer.mark_replay_selected(int(episode_id))
            except (TypeError, ValueError):
                pass
        strength = self._selected_action_strength(selected_action)
        labels = self._labels_from_memory_snapshot(snapshot, limit=max(2, int(limit)))
        feedback_items = [
            dict(item)
            for item in list(snapshot.get("action_feedback_items", []) or snapshot.get("items", []) or [])
            if isinstance(item, dict)
            and (
                str(item.get("sa_label", "") or "").startswith("action_feedback::")
                or str(item.get("family", "") or "") == "action_feedback"
                or str(item.get("source_type", "") or "") == "action_feedback"
            )
        ][:4]
        feedback_summary = self._summarize_replay_feedback(snapshot=snapshot, feedback_items=feedback_items)
        safety_review_hint = {
            "schema_id": "episode_replay_safety_review_hint/v1",
            "source_memory_id": source_memory_id,
            "risk": round(max(float(source.get("risk", 0.0) or 0.0), float(feedback_summary.get("risk", 0.0) or 0.0)), 4),
            "punishment": round(float(feedback_summary.get("punishment", 0.0) or 0.0), 4),
            "pressure": round(float(feedback_summary.get("pressure", 0.0) or 0.0), 4),
            "requires_external_review": bool(float(feedback_summary.get("risk", 0.0) or 0.0) >= 0.18 or float(source.get("risk", 0.0) or 0.0) >= 0.24),
        }
        meta = {
            "schema_id": "episode_replay_control/v1",
            "action_id": "action::replay_episode",
            "control_kind": "replay_episode",
            "source_action_id": "action::replay_episode",
            "source_memory_id": source_memory_id,
            "source_memory_kind": str(snapshot.get("memory_kind", "") or ""),
            "source_tick_index": int(snapshot.get("tick_index", -1) or -1),
            "source_reason": str(source.get("reason", "") or ""),
            "replayed_labels": labels,
            "feedback_summary": feedback_summary,
            "safety_review_hint": safety_review_hint,
            "strength": round(strength, 4),
            "learning_boundary": "episode_replay_can_shape_action_consequence_but_not_concept_embedding",
        }
        rows: list[dict] = [
            {
                "sa_label": "control::episode_replay",
                "display_text": "经验回放",
                "family": "action_control",
                "source_type": "action_control",
                "real_energy": 0.0,
                "virtual_energy": round(min(0.76, 0.20 + strength * 0.48 + float(feedback_summary.get("risk", 0.0) or 0.0) * 0.16), 4),
                "anchor_meta": meta,
            }
        ]
        for label in labels[: max(1, int(limit))]:
            rows.append(
                {
                    "sa_label": label,
                    "display_text": label,
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": round(min(0.62, 0.12 + strength * 0.34 + float(feedback_summary.get("risk", 0.0) or 0.0) * 0.08), 4),
                    "anchor_meta": {**meta, "target_label": label, "target_modulation": "episode_replay"},
                }
            )
        for item in feedback_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            rows.append(
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "family": "action_control",
                    "source_type": "action_control",
                    "real_energy": 0.0,
                    "virtual_energy": 0.0,
                    "attention_gain": round(min(0.52, 0.10 + strength * 0.28), 4),
                    "anchor_meta": {**meta, "target_label": label, "target_modulation": "episode_feedback_replay"},
                }
            )
        return rows

    def _wait_control_row(self, *, selected_action: dict) -> dict:
        strength = self._selected_action_strength(selected_action)
        params = dict((selected_action or {}).get("params", {}) or {})
        duration = max(1, int(params.get("duration_ticks", 1) or 1))
        meta = {
            "schema_id": "timing_wait_control/v1",
            "action_id": "action::wait",
            "control_kind": "wait",
            "source_action_id": "action::wait",
            "wait_hold_ticks": duration,
            "wait_intensity": round(strength, 4),
            "rhythm_expectation": round(float(params.get("rhythm_expectation", 0.0) or 0.0), 4),
            "uncertainty": round(float(params.get("uncertainty", 0.0) or 0.0), 4),
            "external_action_review_hint": round(min(0.65, 0.10 + strength * 0.42), 4),
            "meaning": "legal_non_action_that_can_be_rewarded_or_punished",
        }
        return {
            "sa_label": "control::timing_wait",
            "display_text": "等待",
            "family": "action_control",
            "source_type": "action_control",
            "real_energy": 0.0,
            "virtual_energy": round(min(0.58, 0.12 + strength * 0.35), 4),
            "anchor_meta": meta,
        }

    def _select_episode_replay_source(self, *, selected_action: dict, expectation_pressure_trace: dict, action_consequence_trace: dict) -> dict:
        params = dict((selected_action or {}).get("params", {}) or {})
        explicit = str(params.get("source_memory_id", "") or "")
        if explicit:
            return {"source_memory_id": explicit, "reason": "selected_action_param", "risk": float(params.get("risk", 0.0) or 0.0)}
        anchors = [
            dict(anchor)
            for anchor in list(((expectation_pressure_trace or {}).get("anchor_verification", {}) or {}).get("anchors", []) or [])
            if isinstance(anchor, dict) and str(anchor.get("source_memory_id", "") or "")
        ]
        anchors.sort(
            key=lambda anchor: (
                0 if str(anchor.get("anchor_type", "") or "") == "pressure" else 1,
                -float(anchor.get("level", 0.0) or 0.0),
                -float(anchor.get("expected_punishment", 0.0) or 0.0),
            )
        )
        if anchors:
            anchor = anchors[0]
            risk = float(anchor.get("level", 0.0) or 0.0) * 0.58 + float(anchor.get("expected_punishment", 0.0) or 0.0) * 0.32
            return {"source_memory_id": str(anchor.get("source_memory_id", "") or ""), "reason": "pressure_b_anchor", "risk": risk}
        estimates = list(dict((action_consequence_trace or {}).get("action_estimates", {}) or {}).values())
        estimates = [dict(row) for row in estimates if isinstance(row, dict) and list(row.get("source_memory_ids", []) or [])]
        estimates.sort(
            key=lambda row: (
                -(float(row.get("support", 0.0) or 0.0) * (float(row.get("punishment", 0.0) or 0.0) + float(row.get("pressure", 0.0) or 0.0))),
                str(row.get("action_id", "") or ""),
            )
        )
        if estimates:
            row = estimates[0]
            source_ids = list(row.get("source_memory_ids", []) or [])
            return {
                "source_memory_id": str(source_ids[0] if source_ids else ""),
                "reason": "action_consequence_evidence",
                "risk": float(row.get("support", 0.0) or 0.0) * (float(row.get("punishment", 0.0) or 0.0) + float(row.get("pressure", 0.0) or 0.0)),
            }
        latest = self.memory.latest_snapshot("state")
        return {
            "source_memory_id": str((latest or {}).get("memory_id", "") or ""),
            "reason": "latest_state_fallback",
            "risk": 0.0,
        }

    def _summarize_replay_feedback(self, *, snapshot: dict, feedback_items: list[dict]) -> dict:
        reward = 0.0
        punishment = 0.0
        correctness = 0.0
        pressure = 0.0
        inhibition_count = 0
        labels = []
        for item in list(feedback_items or []) + list((snapshot or {}).get("items", []) or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            if label.startswith("action_inhibition::"):
                inhibition_count += 1
                pressure += float(item.get("real_energy", 0.0) or 0.0) * 0.42
            if label.startswith("signal::punishment"):
                punishment += float(item.get("real_energy", 0.0) or 0.0) + float(item.get("virtual_energy", 0.0) or 0.0) * 0.35
            if label.startswith("signal::reward"):
                reward += float(item.get("real_energy", 0.0) or 0.0)
            if label.startswith("signal::correctness"):
                correctness += float(item.get("real_energy", 0.0) or 0.0)
            if label.startswith("action_feedback::"):
                labels.append(label)
                meta = dict(item.get("anchor_meta", {}) or {})
                observed = dict(meta.get("observed_feedback", {}) or {})
                reward += float(observed.get("reward", 0.0) or 0.0)
                punishment += float(observed.get("punishment", 0.0) or 0.0)
                correctness += float(observed.get("correctness", 0.0) or 0.0)
                semantics = dict(meta.get("feedback_energy_semantics", {}) or {})
                pressure += float(semantics.get("punishment_pressure", 0.0) or 0.0)
        risk = max(0.0, punishment * 0.62 + pressure * 0.38 - reward * 0.16 - correctness * 0.08)
        return {
            "schema_id": "episode_replay_feedback_summary/v1",
            "reward": round(reward, 4),
            "punishment": round(punishment, 4),
            "correctness": round(correctness, 4),
            "pressure": round(pressure, 4),
            "risk": round(risk, 4),
            "inhibition_count": int(inhibition_count),
            "feedback_labels": labels[:8],
        }

    def _selected_action_strength(self, selected_action: dict) -> float:
        decisiveness = float((selected_action or {}).get("effective_decisiveness", 0.0) or 0.0)
        drive = float((selected_action or {}).get("drive", 0.0) or 0.0)
        threshold = float((selected_action or {}).get("effective_threshold", 0.0) or 0.0)
        if decisiveness <= 0.0 and drive > threshold:
            decisiveness = drive - threshold
        innate_strength = max(
            [
                float(node.get("strength", 0.0) or 0.0)
                for node in list((selected_action or {}).get("innate_nodes", []) or [])
                if isinstance(node, dict)
            ]
            or [0.0]
        )
        return max(0.08, min(0.92, 0.18 + decisiveness * 0.55 + innate_strength * 0.12))

    def _labels_from_memory_snapshot(self, snapshot: dict, *, limit: int = 12) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for item in list((snapshot or {}).get("state_field_items", []) or (snapshot or {}).get("core_items", []) or (snapshot or {}).get("items", []) or []):
            if len(labels) >= max(1, int(limit)):
                break
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen or not self._is_core_trace_label(label):
                continue
            seen.add(label)
            labels.append(label)
        if labels:
            return labels
        for label in list((snapshot or {}).get("focus_labels", []) or []):
            clean = str(label or "")
            if clean and clean not in seen and self._is_core_trace_label(clean):
                seen.add(clean)
                labels.append(clean)
                if len(labels) >= max(1, int(limit)):
                    break
        return labels

    def _labels_after_action_control(self, *, feedback_focus_rows: list[dict], control_items: list[dict], action_items: list[dict]) -> list[str]:
        labels = []
        seen = set()
        for row in list(control_items or []) + list(action_items or []) + list(feedback_focus_rows or []):
            label = str((row or {}).get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if len(labels) >= 8:
                break
        return labels

    def _build_action_causal_window(
        self,
        *,
        selected_actions: list[dict],
        state_snapshot_before_action: dict,
        feedback_context: dict,
        control_items: list[dict],
        action_items: list[dict],
        text_output_trace: dict,
        state_snapshot_after_output: dict,
    ) -> dict:
        action_ids = [str(row.get("action_id", "") or "") for row in (selected_actions or []) if str(row.get("action_id", "") or "")]

        def _labels(snapshot: dict, limit: int = 10) -> list[str]:
            return [
                str(item.get("sa_label", "") or "")
                for item in (snapshot.get("items", []) or [])[:limit]
                if str(item.get("sa_label", "") or "")
            ]

        before_labels = _labels(state_snapshot_before_action)
        after_labels = _labels(state_snapshot_after_output)
        output_events = list((text_output_trace or {}).get("recent_events", []) or [])
        revision_events = list((text_output_trace or {}).get("revision_events", []) or [])
        text_parameter_events = self.text_actuator.parameter_events(output_events)
        output_item_labels = [
            str(item.get("sa_label", "") or "")
            for item in (text_output_trace or {}).get("output_items", []) or []
            if str(item.get("sa_label", "") or "")
        ]
        return {
            "schema_id": "action_causal_window/v1",
            "tick_index": int(self.tick_index),
            "action_ids": action_ids,
            "before_top_labels": before_labels,
            "after_top_labels": after_labels,
            "entered_labels": [label for label in after_labels if label not in set(before_labels)][:8],
            "control_labels": [str(item.get("sa_label", "") or "") for item in (control_items or [])[:8]],
            "action_control_effects": [
                dict((item.get("anchor_meta", {}) or {}))
                for item in (control_items or [])[:12]
                if str((item.get("anchor_meta", {}) or {}).get("schema_id", "") or "").endswith("_control/v1")
            ],
            "visual_gaze_events": [
                dict(event)
                for event in (feedback_context or {}).get("visual_gaze_events", []) or []
                if isinstance(event, dict)
            ][:8],
            "auditory_band_events": [
                dict(event)
                for event in (feedback_context or {}).get("auditory_band_events", []) or []
                if isinstance(event, dict)
            ][:8],
            "text_parameter_events": [
                dict(event)
                for event in text_parameter_events
                if isinstance(event, dict)
            ][:8],
            "visual_gaze_state": self.visual_gaze_actuator.state(),
            "auditory_band_state": self.auditory_band_actuator.state(),
            "action_item_labels": [str(item.get("sa_label", "") or "") for item in (action_items or [])[:8]],
            "focus_labels_after_control": list((feedback_context or {}).get("focus_labels_after_control", []) or [])[:8],
            "top_labels_after_control": list((feedback_context or {}).get("top_labels_after_control", []) or [])[:8],
            "text_output": {
                "visible_text": str((text_output_trace or {}).get("visible_text", "") or ""),
                "expected_token": str((text_output_trace or {}).get("expected_token", "") or ""),
                "revision_detected": bool((text_output_trace or {}).get("revision_detected", False)),
                "output_item_labels": output_item_labels[:8],
                "recent_events": output_events[:6],
                "revision_events": revision_events[:6],
            },
        }

    def _observe_action_feedback(self, *, selected_actions: list[dict], feedback_context: dict) -> dict:
        top_labels = [str(label or "") for label in (feedback_context.get("top_labels_after_control", []) or []) if str(label or "")]
        focus_labels = [str(label or "") for label in (feedback_context.get("focus_labels_after_control", []) or []) if str(label or "")]
        reward = 0.0
        punishment = 0.0
        correctness = 0.0
        confidence = 0.24
        notes = []
        for row in selected_actions:
            action_id = str(row.get("action_id", "") or "")
            predicted = dict(row.get("predicted_outcome", {}) or {})
            confidence = max(confidence, float(predicted.get("confidence", 0.0) or 0.0))
            if action_id == "action::continue_focus":
                matched = len([label for label in focus_labels if label in top_labels[:5]])
                reward += 0.18 + matched * 0.05
                correctness += 0.16 + matched * 0.06
                punishment += 0.04 if matched == 0 else 0.0
                notes.append("continue_focus_alignment")
            elif action_id == "action::inspect_residual":
                mismatch_labels = [label for label in top_labels if label.startswith("feeling::dissonance") or label.startswith("feeling::surprise")]
                reward += 0.08 + min(0.12, len(mismatch_labels) * 0.04)
                correctness += 0.06
                punishment += 0.12 + min(0.18, max(0, len(top_labels) - 3) * 0.03)
                notes.append("residual_probe_cost")
            elif action_id == "action::replay_recent_context":
                text_hits = len([label for label in top_labels[:6] if label.startswith("text::") or label.startswith("phrase::")])
                reward += 0.22 + text_hits * 0.04
                correctness += 0.18 + text_hits * 0.03
                punishment += 0.03
                notes.append("replay_context_recovery")
            elif action_id == "action::recall_by_expectation":
                anchor_labels = [
                    label
                    for label in top_labels[:8]
                    if not label.startswith(("action::", "action_feedback::", "expectation_pressure::", "feeling::"))
                ]
                pressure_anchor = any(str(anchor.get("anchor_type", "") or "") == "pressure" for anchor in list(row.get("supporting_anchors", []) or []))
                reward += 0.12 + min(0.18, len(anchor_labels) * 0.035)
                correctness += 0.10 + min(0.18, len(anchor_labels) * 0.03)
                punishment += 0.08 if pressure_anchor else 0.035
                notes.append("expectation_anchor_recall")
            elif action_id in {"action::move_gaze_to", "action::nudge_gaze", "action::scan_visual_field", "action::hold_gaze"}:
                reward += 0.10 + min(0.12, len([label for label in top_labels[:8] if label.startswith("vision::")]) * 0.04)
                correctness += 0.08
                punishment += 0.03
                notes.append("visual_gaze_control")
            elif action_id in {"action::zoom_visual_focus", "action::widen_visual_focus"}:
                reward += 0.09
                correctness += 0.07
                punishment += 0.035
                notes.append("visual_focus_scale_control")
            elif action_id in {"action::slide_audio_band", "action::lock_audio_band", "action::narrow_audio_band", "action::widen_audio_band"}:
                reward += 0.10 + min(0.12, len([label for label in top_labels[:8] if label.startswith("audio::")]) * 0.04)
                correctness += 0.08
                punishment += 0.035
                notes.append("auditory_band_control")
            elif action_id == "action::stabilize_prediction":
                predicted_mass = len([label for label in top_labels[:6] if label.startswith("text::") or label.startswith("phrase::")])
                reward += 0.16 + predicted_mass * 0.03
                correctness += 0.14 + predicted_mass * 0.025
                punishment += 0.05
                notes.append("prediction_stabilization")
            elif action_id == "action::wait":
                wait_controls = [
                    dict(effect)
                    for effect in list((feedback_context.get("action_control_effects", []) or []))
                    if dict(effect).get("control_kind") == "wait"
                    or dict(effect).get("schema_id") == "timing_wait_control/v1"
                ]
                wait_meta = wait_controls[0] if wait_controls else {}
                wait_intensity = float(wait_meta.get("wait_intensity", row.get("effective_decisiveness", 0.0)) or 0.0)
                rhythm_expectation = float(wait_meta.get("rhythm_expectation", 0.0) or 0.0)
                uncertainty = float(wait_meta.get("uncertainty", 0.0) or 0.0)
                reward += 0.05 + min(0.12, rhythm_expectation * 0.08 + uncertainty * 0.07 + wait_intensity * 0.06)
                punishment += max(0.015, 0.04 - rhythm_expectation * 0.012)
                correctness += 0.025 + min(0.10, rhythm_expectation * 0.06 + uncertainty * 0.04)
                notes.append("timing_wait_semantics")
                if rhythm_expectation > 0.0:
                    notes.append("wait_rhythm_phase_guard")
                if uncertainty > 0.0:
                    notes.append("wait_uncertainty_buffer")
        return {
            "reward": round(reward, 4),
            "punishment": round(punishment, 4),
            "correctness": round(correctness, 4),
            "confidence": round(min(1.0, confidence), 4),
            "notes": notes,
        }

    def _build_action_feedback_items(self, *, selected_actions: list[dict], observed_feedback: dict, planner_feedback: dict, causal_window: dict | None = None) -> list[dict]:
        items = []
        reward_energy = float(observed_feedback.get("reward", 0.0) or 0.0)
        punishment_energy = float(observed_feedback.get("punishment", 0.0) or 0.0)
        correctness_energy = float(observed_feedback.get("correctness", 0.0) or 0.0)
        confidence = float(observed_feedback.get("confidence", 0.0) or 0.0)
        pressure_energy = max(0.0, punishment_energy * 0.82 - reward_energy * 0.18 - correctness_energy * 0.08)
        for row in selected_actions:
            action_name = str(row.get("action_id", "") or "").split("::")[-1]
            action_id = str(row.get("action_id", "") or "")
            outcome_estimates = list(((planner_feedback or {}).get("outcome_memory", {}) or {}).get("estimates", []) or [])
            outcome_estimate = next((dict(item) for item in outcome_estimates if str(item.get("action_id", "") or "") == action_id), {})
            items.append(
                {
                    "sa_label": f"action_feedback::{action_name}",
                    "display_text": f"行动反馈:{action_name}",
                    "source_type": "action_feedback",
                    "family": "action_feedback",
                    "real_energy": round(reward_energy + correctness_energy * 0.35, 4),
                    "virtual_energy": round(punishment_energy + pressure_energy * 0.55, 4),
                    "anchor_meta": {
                        "action_id": action_id,
                        "observed_feedback": dict(observed_feedback),
                        "planner_feedback": dict(planner_feedback or {}),
                        "predicted_outcome": dict(row.get("predicted_outcome", {}) or {}),
                        "consequence_estimate": dict(row.get("consequence_estimate", {}) or {}),
                        "outcome_memory_estimate": outcome_estimate,
                        "feedback_energy_semantics": {
                            "schema_id": "action_feedback_energy/v1",
                            "real_energy": round(reward_energy + correctness_energy * 0.35, 4),
                            "virtual_energy": round(punishment_energy + pressure_energy * 0.55, 4),
                            "punishment_pressure": round(pressure_energy, 4),
                            "confidence": round(confidence, 4),
                            "meaning": "reward_correctness_as_real;punishment_pressure_as_virtual_drive_shaping",
                        },
                        "causal_window": dict(causal_window or {}),
                    },
                }
            )
        return items

    def _structured_action_outcome_events(
        self,
        *,
        selected_actions: list[dict],
        observed_feedback: dict,
        planner_feedback: dict,
        parameter_events: list[dict] | None = None,
    ) -> list[dict]:
        events = []
        selected = [dict(row) for row in list(selected_actions or []) if isinstance(row, dict)]
        if not selected and not observed_feedback:
            return []
        reward = float((observed_feedback or {}).get("reward", 0.0) or 0.0)
        punishment = float((observed_feedback or {}).get("punishment", 0.0) or 0.0)
        correctness = float((observed_feedback or {}).get("correctness", 0.0) or 0.0)
        confidence = float((observed_feedback or {}).get("confidence", 0.0) or 0.0)
        if not selected:
            selected = [{"action_id": "action::external_feedback_context", "predicted_outcome": {}}]
        outcome_estimates = list(((planner_feedback or {}).get("outcome_memory", {}) or {}).get("estimates", []) or [])
        for row in selected:
            action_id = str(row.get("action_id", "") or "")
            if not action_id:
                continue
            outcome_estimate = next((dict(item) for item in outcome_estimates if str(item.get("action_id", "") or "") == action_id), {})
            parameter_event_rows = [
                dict(event)
                for event in list(parameter_events or [])
                if isinstance(event, dict) and str(event.get("action_id", "") or "") == action_id
            ]
            parameter_estimates = list((planner_feedback or {}).get("parameter_estimates", []) or [])
            parameter_estimate = next(
                (
                    dict(item)
                    for item in parameter_estimates
                    if isinstance(item, dict) and str(item.get("action_id", "") or "") == action_id
                ),
                {},
            )
            predicted = dict(row.get("predicted_outcome", {}) or {})
            predicted_utility = (
                float(predicted.get("reward", 0.0) or 0.0)
                + float(predicted.get("correctness", 0.0) or 0.0) * 0.42
                - float(predicted.get("punishment", 0.0) or 0.0) * 1.08
                - float(predicted.get("pressure", 0.0) or 0.0) * 0.28
            )
            observed_utility = reward + correctness * 0.42 - punishment * 1.08
            prediction_error = abs(observed_utility - predicted_utility) if predicted else 0.0
            events.append(
                self.learning_event_builder.build(
                    event_type="action_outcome",
                    learning_layer="action_outcome_memory",
                    writer="ActionOutcomeMemory.record",
                    source=action_id,
                    target="observed_feedback",
                    relation="action_outcome",
                    weight=round(min(1.0, reward + punishment + correctness * 0.7 + prediction_error * 0.3), 4),
                    tick_index=self.tick_index,
                    bc_rule_id="BC-004",
                    write_mode="direct_action_outcome_update" if action_id != "action::external_feedback_context" else "feedback_signal_only",
                    evidence={
                        "observed_feedback": dict(observed_feedback or {}),
                        "predicted_outcome": predicted,
                        "observed_utility": round(observed_utility, 4),
                        "predicted_utility": round(predicted_utility, 4),
                        "prediction_error": round(prediction_error, 4),
                        "outcome_memory_estimate": outcome_estimate,
                        "action_params": dict(row.get("params", {}) or {}),
                        "parameter_events": parameter_event_rows[:4],
                        "parameter_memory_estimate": parameter_estimate,
                    },
                    guards=self.learning_event_builder.concept_guards(),
                    meaning="reward and punishment shape action drive; parameterized actuator evidence can shape future action parameters without writing concept similarity",
                )
            )
        return events

    def _extract_replay_labels(self, *, fast_bn: list[dict], slow_bn: list[dict], focus_labels: list[str]) -> list[str]:
        labels = []
        seen = set()
        for label in focus_labels:
            if label and label not in seen and self._is_core_trace_label(label):
                seen.add(label)
                labels.append(label)
        for branch in list(slow_bn) + list(fast_bn):
            snapshot = dict(branch.get("snapshot", {}) or {})
            if snapshot:
                candidate_labels = [
                    str((item or {}).get("sa_label", "") or "")
                    for item in (snapshot.get("state_field_items", []) or snapshot.get("core_items", []) or snapshot.get("items", []) or [])
                ]
            else:
                preview = dict(branch.get("snapshot_preview", {}) or {})
                candidate_labels = [str(label or "") for label in (preview.get("labels", []) or [])]
            for label in candidate_labels:
                if label and label not in seen and self._is_core_trace_label(label):
                    seen.add(label)
                    labels.append(label)
        return labels

    def _extract_predicted_labels(self, *, fast_cn: list[dict], slow_cn: list[dict]) -> list[str]:
        labels = []
        seen = set()
        for branch in list(slow_cn) + list(fast_cn):
            for item in branch.get("predicted_items", []) or []:
                label = str((item or {}).get("sa_label", "") or "")
                if label and label not in seen and self._is_core_trace_label(label):
                    seen.add(label)
                    labels.append(label)
        return labels

    def _is_core_trace_label(self, label: str) -> bool:
        clean = str(label or "")
        if not clean:
            return False
        return True

    def _write_memory_snapshots(
        self,
        state_snapshot: dict,
        source_text: str,
        focus_labels: list[str],
        *,
        asset_refs: list[dict] | None = None,
        state_snapshot_for_memory: dict | None = None,
    ) -> None:
        # IMPORTANT: Memory write must not drop current-tick external evidence.
        # Use a dedicated snapshot view policy for memory write.
        state_snapshot_for_memory = dict(state_snapshot_for_memory or self.state_pool.snapshot_for_memory_write())
        state_asset_refs = self._dedupe_asset_refs(
            list(asset_refs or []) + self._asset_refs_from_items(state_snapshot_for_memory.get("items", []))
        )
        self.memory.write_snapshot(
            tick_index=self.tick_index,
            memory_kind="state",
            items=state_snapshot_for_memory["items"],
            focus_labels=focus_labels,
            source_text=source_text,
            asset_refs=state_asset_refs,
        )
        focus_items = self.state_pool.rows_for_labels(focus_labels)
        focus_asset_refs = self._dedupe_asset_refs(list(asset_refs or []) + self._asset_refs_from_items(focus_items))
        self.memory.write_snapshot(
            tick_index=self.tick_index,
            memory_kind="focus",
            items=focus_items,
            focus_labels=focus_labels,
            source_text=source_text,
            asset_refs=focus_asset_refs,
        )
        slot_trace = getattr(self, "_last_short_term_slot_trace", None)
        slot_items = list((slot_trace or {}).get("items", []) or [])
        if slot_items:
            slot_focus_labels = list((slot_trace or {}).get("focus_labels", []) or [])
            slot_asset_refs = self._dedupe_asset_refs(list(asset_refs or []) + self._asset_refs_from_items(slot_items))
            self.memory.write_snapshot(
                tick_index=self.tick_index,
                memory_kind="short_term_slot",
                items=slot_items,
                focus_labels=slot_focus_labels,
                source_text=source_text,
                asset_refs=slot_asset_refs,
            )

    def _asset_refs_from_items(self, items: list[dict]) -> list[dict]:
        refs: list[dict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            meta = dict(item.get("anchor_meta", {}) or {})
            refs.extend([dict(ref) for ref in list(meta.get("asset_refs", []) or []) if isinstance(ref, dict)])
        return self._dedupe_asset_refs(refs)

    def _build_time_context(self, time_trace: dict) -> dict | None:
        dominant = dict(time_trace.get("dominant_peak", {}) or {})
        items = list(time_trace.get("items", []) or [])
        if not dominant:
            return None
        felt_energy = float(items[0].get("real_energy", 0.0) or 0.0) if items else 0.0
        return {
            "current_tick": self.tick_index,
            "target_delta_t": dominant.get("center_delta_t"),
            "time_sigma": max(1.0, float(dominant.get("sigma", 1.0) or 1.0)),
            "confidence": float(dominant.get("confidence", 0.0) or 0.0),
            "gain": self.config.time_feeling.recall_gain,
            "felt_energy": felt_energy,
        }

    def _merge_feelings_with_expectation_pressure(self, feeling_trace: dict, expectation_pressure_trace: dict) -> dict:
        merged = dict(feeling_trace or {})
        channels = dict(merged.get("channels", {}) or {})
        ep_channels = dict((expectation_pressure_trace or {}).get("channels", {}) or {})
        if ep_channels:
            channels["expectation"] = max(
                float(channels.get("expectation", 0.0) or 0.0),
                float(ep_channels.get("expectation_level", 0.0) or 0.0),
            )
            channels["pressure"] = max(
                float(channels.get("pressure", 0.0) or 0.0),
                float(ep_channels.get("pressure_level", 0.0) or 0.0),
            )
            channels["correctness"] = max(
                float(channels.get("correctness", 0.0) or 0.0),
                float(ep_channels.get("satisfaction_level", 0.0) or 0.0) * 0.62,
            )
        merged["channels"] = channels
        merged["expectation_pressure_coupling"] = {
            "expectation_level": float(ep_channels.get("expectation_level", 0.0) or 0.0),
            "pressure_level": float(ep_channels.get("pressure_level", 0.0) or 0.0),
            "satisfaction_level": float(ep_channels.get("satisfaction_level", 0.0) or 0.0),
            "expectation_gap": float(ep_channels.get("expectation_gap", 0.0) or 0.0),
        }
        return merged

    def _merge_feelings_with_runtime_load(self, feeling_trace: dict, runtime_load_trace: dict) -> dict:
        merged = dict(feeling_trace or {})
        channels = dict(merged.get("channels", {}) or {})
        load_channels = dict((runtime_load_trace or {}).get("channels", {}) or {})
        if load_channels:
            channels["complexity"] = max(
                float(channels.get("complexity", 0.0) or 0.0),
                float(load_channels.get("complexity", 0.0) or 0.0),
            )
            channels["simplicity"] = max(
                float(channels.get("simplicity", 0.0) or 0.0),
                float(load_channels.get("simplicity", 0.0) or 0.0),
            )
        merged["channels"] = channels
        existing_items = list(merged.get("items", []) or [])
        load_items = list((runtime_load_trace or {}).get("items", []) or [])
        if load_items:
            seen = {str((item or {}).get("sa_label", "") or "") for item in existing_items if isinstance(item, dict)}
            for item in load_items:
                label = str((item or {}).get("sa_label", "") or "")
                if label and label not in seen:
                    existing_items.append(dict(item))
                    seen.add(label)
        merged["items"] = existing_items
        merged["runtime_load_coupling"] = {
            "schema_id": "runtime_load_feeling_coupling/v1",
            "complexity": float(load_channels.get("complexity", 0.0) or 0.0),
            "simplicity": float(load_channels.get("simplicity", 0.0) or 0.0),
            "load_ratio": float(load_channels.get("load_ratio", 0.0) or 0.0),
            "suggested_modulation": dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {}),
        }
        return merged

    def _merge_feelings_with_task_feeling(self, feeling_trace: dict, task_feeling_trace: dict) -> dict:
        merged = dict(feeling_trace or {})
        channels = dict(merged.get("channels", {}) or {})
        task_channels = dict((task_feeling_trace or {}).get("channels", {}) or {})
        for key in ("boredom", "fulfillment", "task_available", "unfinished_strength"):
            if key in task_channels:
                channels[key] = max(float(channels.get(key, 0.0) or 0.0), float(task_channels.get(key, 0.0) or 0.0))
        merged["channels"] = channels
        existing_items = list(merged.get("items", []) or [])
        task_items = list((task_feeling_trace or {}).get("items", []) or [])
        if task_items:
            seen = {str((item or {}).get("sa_label", "") or "") for item in existing_items if isinstance(item, dict)}
            for item in task_items:
                label = str((item or {}).get("sa_label", "") or "")
                if label and label not in seen:
                    existing_items.append(dict(item))
                    seen.add(label)
        merged["items"] = existing_items
        merged["task_feeling_coupling"] = {
            "schema_id": "task_feeling_coupling/v1",
            "channels": task_channels,
            "policy": str((task_feeling_trace or {}).get("policy", "") or ""),
        }
        return merged

    def _build_thought_view(
        self,
        *,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        attention_trace: dict,
        feeling_trace: dict,
        runtime_load_trace: dict | None = None,
        focus_continuation_trace: dict | None = None,
        expectation_pressure_trace: dict | None = None,
        text_output_trace: dict | None = None,
    ) -> dict:
        return {
            "fast": {
                "bn": fast_bn,
                "cn": fast_cn,
            },
            "slow": {
                "bn_prime": slow_bn,
                "cn_prime": slow_cn,
            },
            "focus_reason": {
                "selected_labels": list(attention_trace.get("selected_labels", []) or []),
                "ranked_items": list(attention_trace.get("ranked_items", []) or [])[:5],
                "continuation": self._compact_focus_continuation_trace(focus_continuation_trace or {}),
                "focus_order": dict(attention_trace.get("focus_order", {}) or {}),
            },
            "feelings": {
                "channels": dict(feeling_trace.get("channels", {}) or {}),
                "items": list(feeling_trace.get("items", []) or []),
                "prediction_coupling": dict(feeling_trace.get("prediction_coupling", {}) or {}),
            },
            "runtime_load": {
                "channels": dict((runtime_load_trace or {}).get("channels", {}) or {}),
                "components": dict((runtime_load_trace or {}).get("components", {}) or {}),
                "suggested_modulation": dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {}),
            },
            "expectation_pressure": {
                "channels": dict((expectation_pressure_trace or {}).get("channels", {}) or {}),
                "field_state": dict((expectation_pressure_trace or {}).get("field_state", {}) or {}),
                "items": list((expectation_pressure_trace or {}).get("items", []) or []),
                "anchor_verification": dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {}),
            },
            "text_output": {
                "visible_text": str((text_output_trace or {}).get("visible_text", "") or ""),
                "recent_events": list((text_output_trace or {}).get("recent_events", []) or [])[:6],
                "revision_events": list((text_output_trace or {}).get("revision_events", []) or [])[:6],
            },
        }

    def _build_runtime_thought_refs(
        self,
        *,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        attention_trace: dict,
        feeling_trace: dict,
        runtime_load_trace: dict | None = None,
        focus_continuation_trace: dict | None = None,
        expectation_pressure_trace: dict | None = None,
        text_output_trace: dict | None = None,
    ) -> dict:
        return {
            "mode": "runtime_refs",
            "rebuild_policy": "observatory_reconstructs_details_by_snapshot_ref",
            "fast": {
                "bn_refs": [dict(row.get("snapshot_ref", {}) or {}) for row in list(fast_bn or [])[:4]],
                "cn_refs": [self._cn_ref(row) for row in list(fast_cn or [])[:4]],
            },
            "slow": {
                "bn_prime_refs": [dict(row.get("snapshot_ref", {}) or {}) for row in list(slow_bn or [])[:4]],
                "cn_prime_refs": [self._cn_ref(row) for row in list(slow_cn or [])[:4]],
            },
            "focus_reason": {
                "selected_labels": list(attention_trace.get("selected_labels", []) or []),
                "focus_order": dict(attention_trace.get("focus_order", {}) or {}),
                "active_episode_id": int((focus_continuation_trace or {}).get("active_episode_id", -1) or -1),
                "replay_candidates": list((focus_continuation_trace or {}).get("replay_candidates", []) or [])[:4],
            },
            "feelings": {
                "channels": dict(feeling_trace.get("channels", {}) or {}),
            },
            "runtime_load": {
                "channels": dict((runtime_load_trace or {}).get("channels", {}) or {}),
                "suggested_modulation": dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {}),
            },
            "expectation_pressure": {
                "channels": dict((expectation_pressure_trace or {}).get("channels", {}) or {}),
                "field_state": dict((expectation_pressure_trace or {}).get("field_state", {}) or {}),
                "anchor_verification": dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {}),
            },
            "text_output": {
                "visible_text": str((text_output_trace or {}).get("visible_text", "") or ""),
                "revision_detected": bool((text_output_trace or {}).get("revision_detected", False)),
            },
        }

    def _build_runtime_explainability_refs(
        self,
        *,
        state_snapshot: dict,
        fast_bn: list[dict],
        slow_bn: list[dict],
        attention_trace: dict,
        feeling_trace: dict,
        runtime_load_trace: dict | None,
        expectation_pressure_trace: dict | None,
        action_trace: dict,
        action_consequence_trace: dict | None,
        emotion_update_trace: dict | None,
        emotion_modulation: dict | None,
        prior_emotion_modulation: dict | None,
        text_output_trace: dict | None,
        focus_continuation_trace: dict | None = None,
        innate_traces: dict | None = None,
    ) -> dict:
        state_items = list(state_snapshot.get("items", []) or [])[:6]
        return {
            "mode": "runtime_refs",
            "rebuild_policy": "full_whitebox_is_rebuilt_after_tick_from_snapshot_ref_and_memory_store",
            "state_pool": {
                "top_energy_rows": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                        "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                        "cognitive_pressure": float(item.get("cognitive_pressure", 0.0) or 0.0),
                    }
                    for item in state_items
                ],
                "prediction_trace": dict(state_snapshot.get("prediction_trace", {}) or self.state_pool.prediction_trace()),
                "residual_summary": dict(state_snapshot.get("residual_summary", {}) or self.state_pool.residual_summary(limit=8)),
                "energy_flow": dict(state_snapshot.get("energy_flow", {}) or {}),
            },
            "focus": {
                "selected_labels": list(attention_trace.get("selected_labels", []) or []),
                "focus_order": dict(attention_trace.get("focus_order", {}) or {}),
                "continuation": self._compact_focus_continuation_trace(focus_continuation_trace or {}),
            },
            "fast_bn": [self._bn_ref(row) for row in list(fast_bn or [])[:4]],
            "slow_bn": [self._bn_ref(row) for row in list(slow_bn or [])[:4]],
            "feelings": {
                "channels": dict(feeling_trace.get("channels", {}) or {}),
                "prediction_coupling": dict(feeling_trace.get("prediction_coupling", {}) or {}),
            },
            "runtime_load": {
                "channels": dict((runtime_load_trace or {}).get("channels", {}) or {}),
                "components": dict((runtime_load_trace or {}).get("components", {}) or {}),
                "suggested_modulation": dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {}),
            },
            "expectation_pressure": {
                "channels": dict((expectation_pressure_trace or {}).get("channels", {}) or {}),
                "field_state": dict((expectation_pressure_trace or {}).get("field_state", {}) or {}),
                "anchor_verification": dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {}),
            },
            "emotion": {
                "state": dict((emotion_update_trace or {}).get("emotion_state", {}) or {}),
                "cfs_deltas": dict((emotion_update_trace or {}).get("cfs_deltas", {}) or {}),
                "rwd_pun_deltas": dict((emotion_update_trace or {}).get("rwd_pun_deltas", {}) or {}),
                "innate_deltas": dict((emotion_update_trace or {}).get("innate_deltas", {}) or {}),
                "modulation": dict(emotion_modulation or {}),
                "prior_attention_modulation": dict((prior_emotion_modulation or {}).get("attention", {}) or {}),
            },
            "innate_rules": self._compact_innate_traces(innate_traces or {}),
            "action": {
                "consequence_trace": dict(action_consequence_trace or {}),
                "competition_trace": dict(action_trace.get("competition_trace", {}) or {}),
                "safety_gate": dict(action_trace.get("safety_gate", {}) or {}),
                "visual_gaze": dict(action_trace.get("visual_gaze", {}) or {}),
                "auditory_band": dict(action_trace.get("auditory_band", {}) or {}),
                "selected_actions": [
                    {
                        "action_id": str(item.get("action_id", "") or ""),
                        "drive": float(item.get("drive", 0.0) or 0.0),
                        "utility": float(item.get("utility", 0.0) or 0.0),
                        "predicted_outcome": dict(item.get("predicted_outcome", {}) or {}),
                        "consequence_estimate": dict(item.get("consequence_estimate", {}) or {}),
                    }
                    for item in list(action_trace.get("selected_actions", []) or [])[:4]
                ],
            },
            "text_output": {
                "visible_text": str((text_output_trace or {}).get("visible_text", "") or ""),
                "revision_detected": bool((text_output_trace or {}).get("revision_detected", False)),
            },
        }

    def _bn_ref(self, row: dict) -> dict:
        score_breakdown = dict((row or {}).get("score_breakdown", {}) or {})
        numeric_components = []
        for name, value in score_breakdown.items():
            try:
                numeric_components.append((str(name), float(value or 0.0)))
            except (TypeError, ValueError):
                continue
        components = sorted(numeric_components, key=lambda item: (-float(item[1] or 0.0), item[0]))
        return {
            "memory_id": str((row or {}).get("memory_id", "") or ""),
            "tick_index": int((row or {}).get("tick_index", -1) or -1),
            "memory_kind": str((row or {}).get("memory_kind", "") or ""),
            "score": float((row or {}).get("score", 0.0) or 0.0),
            "normalized_weight": float((row or {}).get("normalized_weight", 0.0) or 0.0),
            "match_efficiency": float((row or {}).get("match_efficiency", 0.0) or 0.0),
            "grasp_confidence": float((row or {}).get("grasp_confidence", 0.0) or 0.0),
            "b_real_energy": float((row or {}).get("b_real_energy", 0.0) or 0.0),
            "b_virtual_energy": float((row or {}).get("b_virtual_energy", 0.0) or 0.0),
            "b_effective_real_energy": float((row or {}).get("b_effective_real_energy", 0.0) or 0.0),
            "b_effective_virtual_energy": float((row or {}).get("b_effective_virtual_energy", 0.0) or 0.0),
            "snapshot_ref": dict((row or {}).get("snapshot_ref", {}) or {}),
            "snapshot_preview": dict((row or {}).get("snapshot_preview", {}) or {}),
            "candidate_sources": list((row or {}).get("candidate_sources", []) or []),
            "top_score_components": [{"name": name, "value": value} for name, value in components[:5]],
            "relative_relation_score": float((row or {}).get("relative_relation_score", 0.0) or 0.0),
            "relation_channels": dict((row or {}).get("relation_channels", {}) or {}),
            "learned_score": float((row or {}).get("learned_score", 0.0) or 0.0),
        }

    def _cn_ref(self, row: dict) -> dict:
        return {
            "source_memory_id": str((row or {}).get("source_memory_id", "") or ""),
            "successor_memory_id": str((row or {}).get("successor_memory_id", "") or ""),
            "score": float((row or {}).get("score", 0.0) or 0.0),
            "source_b_weight": float((row or {}).get("source_b_weight", 0.0) or 0.0),
            "source_b_match_efficiency": float((row or {}).get("source_b_match_efficiency", 0.0) or 0.0),
            "successor_normalized_weight": float((row or {}).get("successor_normalized_weight", 0.0) or 0.0),
            "energy_transfer_multiplier": float((row or {}).get("energy_transfer_multiplier", 0.0) or 0.0),
            "energy_transfer": dict((row or {}).get("energy_transfer", {}) or {}),
            "predicted_label_count": len((row or {}).get("predicted_items", []) or []),
        }

    def _build_explainability(
        self,
        *,
        state_snapshot: dict,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        attention_trace: dict,
        feeling_trace: dict,
        runtime_load_trace: dict | None,
        time_trace: dict,
        rhythm_trace: dict,
        action_trace: dict,
        action_feedback_trace: dict,
        action_consequence_trace: dict | None = None,
        expectation_pressure_trace: dict | None = None,
        text_output_trace: dict | None = None,
        emotion_update_trace: dict | None = None,
        emotion_modulation: dict | None = None,
        prior_emotion_modulation: dict | None = None,
        focus_continuation_trace: dict | None = None,
        innate_traces: dict | None = None,
    ) -> dict:
        return {
            "state_pool": {
                "top_energy_rows": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                        "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                        "cognitive_pressure": float(item.get("cognitive_pressure", 0.0) or 0.0),
                        "is_focus": bool((item.get("anchor_meta", {}) or {}).get("is_focus", False)),
                    }
                    for item in (state_snapshot.get("items", []) or [])[:6]
                ],
                "prediction_trace": dict(state_snapshot.get("prediction_trace", {}) or self.state_pool.prediction_trace()),
                "residual_summary": dict(state_snapshot.get("residual_summary", {}) or self.state_pool.residual_summary(limit=8)),
            },
            "focus": {
                "selected_labels": list(attention_trace.get("selected_labels", []) or []),
                "focus_order": dict(attention_trace.get("focus_order", {}) or {}),
                "continuation": self._compact_focus_continuation_trace(focus_continuation_trace or {}),
                "ranked_items": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "focus_score": float(item.get("focus_score", 0.0) or 0.0),
                        "continuation_bonus": float(item.get("continuation_bonus", 0.0) or 0.0),
                        "cognitive_pressure": float(item.get("cognitive_pressure", 0.0) or 0.0),
                    }
                    for item in (attention_trace.get("ranked_items", []) or [])[:6]
                ],
            },
            "fast_bn": [self._bn_reason(row) for row in fast_bn[:4]],
            "slow_bn": [self._bn_reason(row) for row in slow_bn[:4]],
            "fast_cn": [self._cn_reason(row) for row in fast_cn[:4]],
            "slow_cn": [self._cn_reason(row) for row in slow_cn[:4]],
            "feelings": {
                "channels": dict(feeling_trace.get("channels", {}) or {}),
                "prediction_coupling": dict(feeling_trace.get("prediction_coupling", {}) or {}),
                "items": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                        "anchor_meta": dict(item.get("anchor_meta", {}) or {}),
                    }
                    for item in (feeling_trace.get("items", []) or [])[:8]
                ],
            },
            "runtime_load": {
                "channels": dict((runtime_load_trace or {}).get("channels", {}) or {}),
                "components": dict((runtime_load_trace or {}).get("components", {}) or {}),
                "items": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                        "anchor_meta": dict(item.get("anchor_meta", {}) or {}),
                    }
                    for item in ((runtime_load_trace or {}).get("items", []) or [])[:4]
                ],
                "suggested_modulation": dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {}),
            },
            "expectation_pressure": {
                "channels": dict((expectation_pressure_trace or {}).get("channels", {}) or {}),
                "field_state": dict((expectation_pressure_trace or {}).get("field_state", {}) or {}),
                "anchor_verification": dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {}),
                "items": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "real_energy": float(item.get("real_energy", 0.0) or 0.0),
                        "anchor_meta": dict(item.get("anchor_meta", {}) or {}),
                    }
                    for item in ((expectation_pressure_trace or {}).get("items", []) or [])[:8]
                ],
            },
            "time": {
                "channels": dict(time_trace.get("channels", {}) or {}),
                "dominant_peak": dict(time_trace.get("dominant_peak", {}) or {}),
            },
            "rhythm": {
                "channels": dict(rhythm_trace.get("channels", {}) or {}),
                "family": dict(rhythm_trace.get("family", {}) or {}),
            },
            "emotion": {
                "state": dict((emotion_update_trace or {}).get("emotion_state", {}) or {}),
                "cfs_deltas": dict((emotion_update_trace or {}).get("cfs_deltas", {}) or {}),
                "rwd_pun_deltas": dict((emotion_update_trace or {}).get("rwd_pun_deltas", {}) or {}),
                "innate_deltas": dict((emotion_update_trace or {}).get("innate_deltas", {}) or {}),
                "modulation": dict(emotion_modulation or {}),
                "prior_attention_modulation": dict((prior_emotion_modulation or {}).get("attention", {}) or {}),
            },
            "innate_rules": self._compact_innate_traces(innate_traces or {}),
            "action": {
                "consequence_trace": dict(action_consequence_trace or action_trace.get("consequence_trace", {}) or {}),
                "competition_trace": dict(action_trace.get("competition_trace", {}) or {}),
                "causal_window": dict(action_trace.get("causal_window", {}) or {}),
                "safety_gate": dict(action_trace.get("safety_gate", {}) or {}),
                "selected_actions": [
                    {
                        "action_id": str(item.get("action_id", "") or ""),
                        "drive": float(item.get("drive", 0.0) or 0.0),
                        "utility": float(item.get("utility", 0.0) or 0.0),
                        "predicted_outcome": dict(item.get("predicted_outcome", {}) or {}),
                        "consequence_estimate": dict(item.get("consequence_estimate", {}) or {}),
                        "notes": list(item.get("notes", []) or []),
                    }
                    for item in (action_trace.get("selected_actions", []) or [])[:4]
                ],
                "top_candidates": [
                    {
                        "action_id": str(item.get("action_id", "") or ""),
                        "drive": float(item.get("drive", 0.0) or 0.0),
                        "utility": float(item.get("utility", 0.0) or 0.0),
                        "bias": float(item.get("bias", 0.0) or 0.0),
                        "fatigue": float(item.get("fatigue", 0.0) or 0.0),
                        "feedback_modulation": float(item.get("feedback_modulation", 1.0) or 1.0),
                        "consequence_estimate": dict(item.get("consequence_estimate", {}) or {}),
                        "notes": list(item.get("notes", []) or [])[:6],
                    }
                    for item in (action_trace.get("candidates", []) or [])[:6]
                ],
                "control_items": [
                    {
                        "sa_label": str(item.get("sa_label", "") or ""),
                        "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                        "control_kind": str((item.get("anchor_meta", {}) or {}).get("control_kind", "") or ""),
                    }
                    for item in (action_trace.get("control_items", []) or [])[:6]
                ],
                "action_control_effects": dict(action_trace.get("action_control_effects", {}) or {}),
                "visual_gaze": dict(action_trace.get("visual_gaze", {}) or {}),
                "auditory_band": dict(action_trace.get("auditory_band", {}) or {}),
            },
            "action_feedback": {
                "applied": bool(action_feedback_trace.get("applied", False)),
                "observed_feedback": dict(action_feedback_trace.get("observed_feedback", {}) or {}),
                "selected_actions": [
                    str(item.get("action_id", "") or "")
                    for item in (action_feedback_trace.get("selected_actions", []) or [])[:4]
                ],
            },
            "text_output": {
                "visible_text": str((text_output_trace or {}).get("visible_text", "") or ""),
                "expected_token": str((text_output_trace or {}).get("expected_token", "") or ""),
                "revision_detected": bool((text_output_trace or {}).get("revision_detected", False)),
                "revision_events": list((text_output_trace or {}).get("revision_events", []) or [])[:8],
            },
        }

    def _bn_reason(self, row: dict) -> dict:
        snapshot = dict(row.get("snapshot", {}) or {})
        snapshot_ref = dict(row.get("snapshot_ref", {}) or {})
        score_breakdown = dict(row.get("score_breakdown", {}) or {})
        numeric_components = []
        for name, value in score_breakdown.items():
            try:
                numeric_components.append((str(name), float(value or 0.0)))
            except (TypeError, ValueError):
                continue
        components = sorted(numeric_components, key=lambda item: (-float(item[1] or 0.0), item[0]))
        return {
            "memory_id": str(row.get("memory_id", "") or ""),
            "tick_index": int(row.get("tick_index", snapshot_ref.get("tick_index", snapshot.get("tick_index", -1))) or -1),
            "source_text": str(row.get("source_text", snapshot_ref.get("source_text", snapshot.get("source_text", ""))) or ""),
            "score": float(row.get("score", 0.0) or 0.0),
            "normalized_weight": float(row.get("normalized_weight", 0.0) or 0.0),
            "match_efficiency": float(row.get("match_efficiency", 0.0) or 0.0),
            "grasp_confidence": float(row.get("grasp_confidence", 0.0) or 0.0),
            "b_real_energy": float(row.get("b_real_energy", 0.0) or 0.0),
            "b_virtual_energy": float(row.get("b_virtual_energy", 0.0) or 0.0),
            "b_effective_real_energy": float(row.get("b_effective_real_energy", 0.0) or 0.0),
            "b_effective_virtual_energy": float(row.get("b_effective_virtual_energy", 0.0) or 0.0),
            "energy_transfer": dict(row.get("energy_transfer", {}) or {}),
            "candidate_sources": list(row.get("candidate_sources", []) or []),
            "matched_tokens": dict(row.get("matched_tokens", {}) or {}),
            "snapshot_ref": snapshot_ref,
            "snapshot_preview": dict(row.get("snapshot_preview", {}) or {}),
            "top_score_components": [{"name": name, "value": value} for name, value in components[:5]],
            "relation_channels": dict(row.get("relation_channels", score_breakdown.get("relation_channels", {})) or {}),
            "relation_matches": list(row.get("relation_matches", []) or [])[:6],
            "learned_contributions": list(row.get("learned_contributions", []) or [])[:6],
        }

    def _cn_reason(self, row: dict) -> dict:
        predicted_items = list(row.get("predicted_items", []) or [])
        return {
            "source_memory_id": str(row.get("source_memory_id", "") or ""),
            "successor_memory_id": str(row.get("successor_memory_id", "") or ""),
            "score": float(row.get("score", 0.0) or 0.0),
            "learned_transition_score": float(row.get("learned_transition_score", 0.0) or 0.0),
            "source_b_weight": float(row.get("source_b_weight", 0.0) or 0.0),
            "source_b_match_efficiency": float(row.get("source_b_match_efficiency", 0.0) or 0.0),
            "successor_normalized_weight": float(row.get("successor_normalized_weight", 0.0) or 0.0),
            "energy_transfer_multiplier": float(row.get("energy_transfer_multiplier", 0.0) or 0.0),
            "energy_transfer": dict(row.get("energy_transfer", {}) or {}),
            "predicted_labels": [str(item.get("sa_label", "") or "") for item in predicted_items[:8]],
            "predicted_energies": [
                {
                    "sa_label": str(item.get("sa_label", "") or ""),
                    "virtual_energy": float(item.get("virtual_energy", 0.0) or 0.0),
                }
                for item in predicted_items[:8]
            ],
            "learned_transition_contributions": list(row.get("learned_transition_contributions", []) or [])[:6],
        }
