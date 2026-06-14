from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextSensorConfig:
    budget_limit: int = 1024
    competition_limit: int = 1024
    dynamic_phrase_min_observations: int = 2
    dynamic_phrase_max_len: int = 3
    dynamic_phrase_scan_budget: int = 256
    dynamic_phrase_emit_budget: int = 32


@dataclass(frozen=True)
class VisionSensorConfig:
    mode: str = "native_numeric"
    max_objects: int = 4
    max_side: int = 160
    preview_side: int = 96
    fallback_to_legacy: bool = True


@dataclass(frozen=True)
class AudioSensorConfig:
    mode: str = "native_numeric"
    max_samples: int = 32768
    band_count: int = 12
    fallback_to_legacy: bool = True


@dataclass(frozen=True)
class MultimodalAssetConfig:
    # Disabled by default: APV2.1 observatory must reconstruct inner vision/audio
    # from state-pool SA/numeric channels, not by replaying raw input assets.
    enabled: bool = False
    max_assets: int = 256
    keep_inline_previews: bool = True
    persist_payloads: bool = False
    asset_root_dir: str = "data/multimodal_assets"
    keep_hot_payloads: bool = True
    preview_retention_ticks: int = 64
    object_proxy_retention_ticks: int = 256
    raw_frame_retention_ticks: int = 256
    focus_tile_retention_ticks: int = 512
    focus_tile_max_count: int = 4
    focus_tile_padding_ratio: float = 0.08
    audio_focus_window_max_bytes: int = 48000


@dataclass(frozen=True)
class StatePoolConfig:
    real_decay: float = 0.9
    virtual_decay: float = 0.86
    attention_gain_decay: float = 0.9
    fatigue_decay: float = 0.82
    prune_threshold: float = 0.045
    # --- Fixed-budget readout (R_state) ---
    # 二期草案硬约束：禁止 tick 内全池遍历。状态池可以很大，但每 tick 参与召回/注意力
    # 的读出必须是固定预算、多头读出。
    r_state_head_limit: int = 7
    # Fast-system Bn should be able to query a 1024-level SA field on normal
    # hardware, but the source must still be bounded R_state heads rather than
    # a full state-pool scan.
    r_state_items_per_head: int = 256
    # 每 tick 的维护预算：用于懒惰衰减/淘汰抽查，避免长跑“永不触达的死项”无限堆积。
    maintenance_budget: int = 48
    # 最近外源证据队列上限（用于 head_recent）。
    recent_external_limit: int = 2048
    # 热锚点候选缓存上限（用于 head_anchor / head_global 的候选池）。
    hot_anchor_limit: int = 2048

    # --- View-layer caps (observability + memory snapshot) ---
    # query_view / attention_view 是“白箱可视化视图”，不是召回主查询体。
    query_limit: int = 8
    # snapshot_limit 仍是“可读 view”上限，不等于状态池总规模，也不等于 R_state 预算。
    snapshot_limit: int = 24
    # memory_snapshot_limit 是长期记忆写入视图上限。状态记忆需要保留 1024 级
    # SA 能量分布，不能被 display snapshot 的小上限裁剪掉。
    memory_snapshot_limit: int = 1024
    prediction_validation_actual_limit: int = 256
    prediction_validation_update_limit: int = 128
    focus_boost: float = 0.3
    focus_fatigue_step: float = 0.18
    # Prediction fatigue is intentionally weaker than focus fatigue. A strongly
    # predicted C* object should remain a real background expectation, but if it
    # keeps dominating several ticks in a row, fatigue lets nearby consequences
    # compete for attention instead of hard-clipping the virtual energy itself.
    prediction_fatigue_enabled: bool = True
    prediction_fatigue_min_mass: float = 0.18
    prediction_fatigue_ratio: float = 0.18
    prediction_fatigue_gain: float = 0.06
    prediction_fatigue_max_step: float = 0.18
    cstar_trace_top_labels: int = 8
    bootstrap_virtual_energy: float = 0.6


@dataclass(frozen=True)
class MemoryConfig:
    recall_top_k: int = 5
    predict_top_k: int = 5
    prediction_energy_scale: float = 0.55
    max_snapshots_per_kind: int = 256
    candidate_limit: int = 256
    core_item_limit: int = 1024
    query_feature_limit: int = 1024
    posting_label_token_limit: int = 256
    posting_display_token_limit: int = 128
    posting_bigram_token_limit: int = 192
    posting_sequence_token_limit: int = 192
    vector_token_limit: int = 512
    scoring_candidate_limit: int = 96
    learned_rerank_limit: int = 16
    state_query_signature_token_limit: int = 256
    numeric_enabled: bool = True
    numeric_dim: int = 64
    numeric_candidate_limit: int = 64
    numeric_top_k_per_channel: int = 24
    numeric_weight: float = 1.15
    relation_enabled: bool = True
    relation_token_limit: int = 256
    relation_event_limit: int = 128
    relation_context_limit: int = 8192
    relation_score_weight: float = 0.68
    relation_focus_score_weight: float = 0.92
    # Temporal applicability is a slow humanlike memory-age modulator. It does
    # not delete old memories and does not filter SA types; it only changes how
    # much current grasp an old all-SA snapshot receives before Bn/Cn energy
    # normalization. Defaults use 1 tick ~= 0.1s, so the long half-life is about
    # 30 days. Tests may override this with tiny windows to simulate months.
    temporal_applicability_enabled: bool = True
    temporal_tick_seconds: float = 0.1
    temporal_fatigue_window_ticks: int = 80
    temporal_fatigue_strength: float = 0.92
    temporal_fatigue_recovery_exponent: float = 1.0
    temporal_recent_gain_window_ticks: int = 864_000
    temporal_recent_gain: float = 0.14
    temporal_long_half_life_ticks: int = 25_920_000
    temporal_floor: float = 0.18
    # Runtime write path policy:
    # snapshot payload + transition links are written immediately; heavier
    # posting/ANN/online-learning index maintenance is processed under this
    # bounded per-tick budget.
    index_jobs_per_tick: int = 1
    index_maintenance_min_remaining_ms: float = 36.0
    index_maintenance_max_ms: float = 24.0
    # Heavy state snapshots (1024+ SA) are not indexed on the realtime path
    # unless explicit idle/debug maintenance asks for them. This preserves the
    # full 1024 memory payload while keeping the tick loop time-bounded.
    idle_heavy_index_jobs: int = 1
    idle_index_maintenance_max_ms: float = 12.0


@dataclass(frozen=True)
class AttentionConfig:
    focus_limit: int = 8
    pressure_gain: float = 0.6
    attention_gain_weight: float = 0.8
    fatigue_weight: float = 0.5
    continuation_bias: float = 0.35
    real_energy_weight: float = 0.0
    virtual_energy_weight: float = 0.25
    focus_family_budget_enabled: bool = True
    focus_family_text_max: int = 4
    focus_family_vision_max: int = 3
    focus_family_audio_max: int = 3
    focus_family_cognitive_feeling_max: int = 2
    focus_family_emotion_max: int = 2
    focus_family_action_max: int = 2
    focus_family_time_max: int = 1
    focus_family_rhythm_max: int = 1
    focus_family_expectation_pressure_max: int = 2
    focus_family_other_max: int = 2
    successor_bias_enabled: bool = True
    successor_bias_gain: float = 0.42
    successor_bias_max: float = 0.48
    successor_bias_top_k: int = 12
    successor_bias_context_limit: int = 2048
    successor_bias_max_successors_per_context: int = 64
    successor_bias_max_context_labels: int = 8
    successor_bias_max_order: int = 3
    successor_bias_per_tick_update_limit: int = 16
    successor_bias_real_threshold: float = 0.08
    successor_bias_decay: float = 0.992
    successor_bias_rescale_threshold: float = 64.0
    successor_bias_rescale_factor: float = 0.5
    successor_bias_min_support: float = 0.18
    successor_bias_entropy_floor: float = 0.28


@dataclass(frozen=True)
class ShortTermConfig:
    focus_history_limit: int = 12
    recency_decay: float = 0.78
    synthetic_query_weight: float = 1.1
    replay_decay: float = 0.72
    replay_query_weight: float = 0.82
    max_replay_items: int = 8
    episode_break_overlap: float = 0.22
    echo_enabled: bool = True
    echo_history_limit: int = 128
    echo_max_age_ticks: int = 8
    echo_decay: float = 0.68
    echo_sensory_gain: float = 0.22
    echo_thought_gain: float = 0.18
    echo_max_energy: float = 0.28
    echo_max_items_per_tick: int = 18
    # P1-J-15: modality-specific echo lifetimes. AP uses 1 tick ~= 0.1s,
    # so visual afterimages must fade within a few ticks while auditory
    # aftersounds may remain for phrase/rhythm integration. These are channel
    # dynamics, not content rules; all SA still remain first-class state items.
    echo_vision_max_age_ticks: int = 4
    echo_vision_decay: float = 0.42
    echo_vision_gain: float = 0.18
    echo_vision_max_energy: float = 0.16
    echo_audio_max_age_ticks: int = 24
    echo_audio_decay: float = 0.82
    echo_audio_gain: float = 0.20
    echo_audio_max_energy: float = 0.22
    echo_text_max_age_ticks: int = 10
    echo_text_decay: float = 0.72
    echo_text_gain: float = 0.18
    echo_text_max_energy: float = 0.18
    echo_thought_max_age_ticks: int = 14
    echo_thought_decay: float = 0.76
    echo_thought_modality_gain: float = 0.15
    echo_thought_max_energy: float = 0.16
    # P1-J-16: active short-term memory is a working-memory window, not a
    # sensory afterimage. It may remember recent thought/sensory/action facts
    # for several seconds, but recall is partial and action-triggered.
    memory_window_enabled: bool = True
    memory_window_history_limit: int = 64
    memory_window_max_age_ticks: int = 48
    memory_window_recency_decay: float = 0.86
    memory_window_fatigue_decay: float = 0.70
    memory_window_fatigue_step: float = 0.45
    memory_window_max_items_per_event: int = 12
    memory_window_recall_limit: int = 8


@dataclass(frozen=True)
class ShortTermSlotConfig:
    enabled: bool = True
    capacity: int = 32
    base_virtual_budget: float = 0.72
    item_real_fraction: float = 0.06
    item_min_virtual: float = 0.02
    item_max_virtual: float = 0.14
    item_rank_decay: float = 0.86
    item_order_decay: float = 0.92
    summary_ratio: float = 0.18
    order_ratio: float = 0.16
    continuity_ratio: float = 0.14
    rhythm_ratio: float = 0.10
    load_floor: float = 0.25
    continuity_gain: float = 0.35
    order_gain: float = 0.28
    rhythm_gain: float = 0.22
    working_memory_fill_limit: int = 8
    focus_merge_limit: int = 32


@dataclass(frozen=True)
class CognitiveFeelingConfig:
    min_activation: float = 0.12
    surprise_gain: float = 1.05
    coherence_gain: float = 0.82
    dissonance_gain: float = 1.0
    correctness_gain: float = 0.9
    grasp_gain: float = 0.88
    expectation_gain: float = 0.78
    pressure_gain: float = 0.95


@dataclass(frozen=True)
class TaskFeelingConfig:
    enabled: bool = True
    min_activation: float = 0.12
    boredom_gain: float = 1.0
    fulfillment_gain: float = 1.0
    unfinished_mark_min_strength: float = 0.18


@dataclass(frozen=True)
class RuntimeLoadFeelingConfig:
    enabled: bool = True
    min_activation: float = 0.08
    complexity_gain: float = 0.85
    simplicity_gain: float = 0.55
    target_load_ratio: float = 1.0
    ideal_load_ratio: float = 0.58
    state_item_soft_limit: int = 1024
    r_state_item_soft_limit: int = 1792
    attention_candidate_soft_limit: int = 256
    pending_index_soft_limit: int = 24
    family_overflow_soft_limit: int = 8
    residual_mass_soft_limit: float = 24.0
    mismatch_weight: float = 0.42
    fatigue_decay: float = 0.86
    fatigue_step: float = 0.08
    fatigue_gain: float = 0.45
    max_energy: float = 0.85


@dataclass(frozen=True)
class RuntimeBudgetControllerConfig:
    enabled: bool = True
    smoothing_alpha: float = 1.0
    readout_min_multiplier: float = 0.72
    readout_max_multiplier: float = 1.08
    attention_candidate_min_multiplier: float = 0.68
    attention_candidate_max_multiplier: float = 1.12
    index_jobs_min_multiplier: float = 0.0
    index_jobs_max_multiplier: float = 1.45
    index_time_min_multiplier: float = 0.35
    index_time_max_multiplier: float = 1.25
    trace_detail_min_multiplier: float = 0.6
    trace_detail_max_multiplier: float = 1.12
    min_r_state_items_per_head: int = 32
    preserve_1024_query_floor: bool = True
    max_extra_index_jobs: int = 1


@dataclass(frozen=True)
class TimeFeelingConfig:
    enabled: bool = True
    threshold: float = 0.22
    gain: float = 0.95
    min_confidence: float = 0.24
    default_radius_ticks: float = 4.0
    recall_gain: float = 0.22
    fatigue_decay: float = 0.82
    fatigue_step: float = 0.18
    fatigue_gain: float = 0.55
    fatigue_max: float = 1.0
    max_sources: int = 6
    # Runtime hot path policy:
    # time feelings enter state/attention/memory immediately, but a second
    # 1024-level fast recall is reserved for explicit high-pressure profiles.
    rerun_recall_confidence_threshold: float = 1.01
    rerun_recall_energy_threshold: float = 1.01


@dataclass(frozen=True)
class RhythmConfig:
    enabled: bool = True
    window: int = 12
    min_hits: int = 3
    min_period: int = 2
    max_period: int = 12
    period_sigma_scale: float = 0.18
    phase_sigma_scale: float = 0.22
    pulse_threshold: float = 0.18
    phase_threshold: float = 0.14
    fatigue_decay: float = 0.82
    fatigue_step: float = 0.12
    fatigue_gain: float = 0.55
    fatigue_max: float = 1.0
    salience_threshold: float = 0.18


@dataclass(frozen=True)
class ExpectationPressureConfig:
    enabled: bool = True
    min_activation: float = 0.1
    expectation_decay: float = 0.9
    pressure_decay: float = 0.88
    satisfaction_decay: float = 0.92
    expectation_gain: float = 0.58
    pressure_gain: float = 0.62
    satisfaction_gain: float = 0.5
    residual_gain: float = 1.0
    feedback_gain: float = 1.0
    anchor_verifier_enabled: bool = True
    anchor_max_anchors: int = 32
    anchor_decay: float = 0.88
    anchor_min_level: float = 0.03
    anchor_min_outcome_virtual: float = 0.045
    anchor_validation_gain: float = 0.62
    anchor_miss_gain: float = 0.34


@dataclass(frozen=True)
class InnateRuleConfig:
    enabled: bool = True
    apply_emit_sa: bool = True
    apply_action_nodes: bool = True
    apply_emotion_deltas: bool = True
    min_fire_strength: float = 0.035
    max_items_per_phase: int = 16
    max_action_nodes_per_phase: int = 12
    max_learning_events_per_phase: int = 24
    innate_action_drive_gain: float = 1.0
    memory_action_virtual_drive_gain: float = 0.28
    safety_gate_enabled: bool = True
    safety_veto_pressure_threshold: float = 0.64
    safety_veto_cor_threshold: float = 0.68
    safety_review_pressure_threshold: float = 0.42
    safety_review_cor_threshold: float = 0.50
    safety_min_external_confidence: float = 0.58


@dataclass(frozen=True)
class ActionConfig:
    enabled: bool = True
    selection_threshold: float = 0.32
    max_selected_actions: int = 4
    fatigue_decay: float = 0.84
    fatigue_step: float = 0.14
    bias_learning_rate: float = 0.28
    bias_gain: float = 0.4
    confidence_gain: float = 0.18
    wait_base_drive: float = 0.18
    consequence_max_successor_rows: int = 12
    consequence_max_evidence_per_action: int = 8
    consequence_max_horizon: int = 3
    consequence_branching: int = 3
    consequence_path_decay: float = 0.72
    outcome_memory_enabled: bool = True
    outcome_memory_learning_rate: float = 0.18
    outcome_memory_decay_per_tick: float = 0.992
    outcome_memory_support_scale: float = 6.0
    outcome_memory_max_drive_bias: float = 0.75


@dataclass(frozen=True)
class OnlineEmbeddingConfig:
    enabled: bool = True
    dim: int = 32
    token_limit: int = 2048
    min_support_to_promote: int = 2
    per_tick_update_limit: int = 8
    scoring_token_limit: int = 256
    learned_weight: float = 0.28
    learned_vector_candidate_weight: float = 4.5
    transition_learned_weight: float = 0.18


@dataclass(frozen=True)
class EmotionConfig:
    """8 通道拟人 NT 系统配置"""
    enabled: bool = True
    # CFS → NT 映射增益
    cfs_gain: float = 1.0
    # Rwd/Pun → NT 映射增益
    rwd_pun_gain: float = 1.0


@dataclass(frozen=True)
class TunerConfig:
    enabled: bool = True
    ema_alpha: float = 0.04
    min_support_ticks: int = 12
    target_prediction_alignment: float = 0.58
    max_normal_pressure: float = 3.5
    target_action_success: float = 0.52
    adjustment_rate: float = 0.025
    rollback_threshold: float = 0.18


@dataclass(frozen=True)
class ObservabilityConfig:
    default_trace_mode: str = "summary"
    target_tick_ms: float = 100.0
    disable_gc_during_tick: bool = True
    idle_gc_collect_generation: int = 0
    trace_item_preview_limit: int = 32
    trace_r_state_item_preview_limit: int = 8
    trace_text_preview_chars: int = 512
    trace_matched_token_preview_limit: int = 12


@dataclass(frozen=True)
class RuntimeConfig:
    text_sensor: TextSensorConfig = field(default_factory=TextSensorConfig)
    vision_sensor: VisionSensorConfig = field(default_factory=VisionSensorConfig)
    audio_sensor: AudioSensorConfig = field(default_factory=AudioSensorConfig)
    multimodal_assets: MultimodalAssetConfig = field(default_factory=MultimodalAssetConfig)
    state_pool: StatePoolConfig = field(default_factory=StatePoolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    short_term: ShortTermConfig = field(default_factory=ShortTermConfig)
    short_term_slot: ShortTermSlotConfig = field(default_factory=ShortTermSlotConfig)
    cognitive_feelings: CognitiveFeelingConfig = field(default_factory=CognitiveFeelingConfig)
    task_feeling: TaskFeelingConfig = field(default_factory=TaskFeelingConfig)
    runtime_load_feeling: RuntimeLoadFeelingConfig = field(default_factory=RuntimeLoadFeelingConfig)
    runtime_budget_controller: RuntimeBudgetControllerConfig = field(default_factory=RuntimeBudgetControllerConfig)
    time_feeling: TimeFeelingConfig = field(default_factory=TimeFeelingConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    expectation_pressure: ExpectationPressureConfig = field(default_factory=ExpectationPressureConfig)
    innate_rules: InnateRuleConfig = field(default_factory=InnateRuleConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    online_embedding: OnlineEmbeddingConfig = field(default_factory=OnlineEmbeddingConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    allow_memory_bootstrap: bool = True
