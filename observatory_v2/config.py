# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema_tools import load_schema, repo_root as _repo_root, validate_or_raise


@dataclass(frozen=True)
class AppConfig:
    schema_id: str
    schema_version: str
    host: str
    port: int
    auto_open_browser: bool
    live_ring_limit: int
    run_chunk_size: int
    default_demo_tick_count: int
    default_demo_tick_interval_ms: int
    text_sensor_budget: int
    text_sensor_fatigue_window: int
    text_sensor_fatigue_threshold: int
    text_sensor_max_suppression: float
    text_sensor_verbatim_window_chars: int
    state_pool_decay: float
    state_pool_prune_threshold: float
    state_pool_recent_queue_limit: int
    state_pool_anchor_cache_limit: int
    state_pool_residual_limit: int
    state_pool_handle_limit: int
    state_pool_residual_unit_limit: int
    state_pool_attention_object_fatigue_decay: float
    state_pool_attention_object_fatigue_step: float
    state_pool_attention_object_fatigue_gain: float
    state_pool_attention_object_fatigue_max: float
    state_pool_attention_object_min_multiplier: float
    memory_store_recent_limit: int
    memory_vector_dim: int
    memory_vector_backend: str
    memory_ann_enabled: bool
    memory_ann_top_k: int
    memory_candidate_limit: int
    memory_spacetime_backend: str
    memory_spacetime_time_bucket_size: int
    memory_spacetime_space_bucket_size: float
    memory_spacetime_time_radius: int
    memory_spacetime_space_radius: float
    memory_recall_fatigue_decay: float
    memory_recall_fatigue_gain: float
    memory_recall_fatigue_accumulate_scale: float
    memory_recall_fatigue_max: float
    memory_recall_fatigue_min_multiplier: float
    short_term_memory_limit: int
    short_term_successor_tail_limit: int
    vision_patch_budget: int
    vision_focus_patch_budget: int
    vision_raw_state_budget: int
    vision_reconstruction_patch_budget: int
    vision_edge_candidate_gain: float
    vision_edge_priority_gain: float
    vision_attention_boost_enabled: bool
    vision_attention_boost_decay: float
    vision_attention_boost_max_extra_raw_budget: int
    vision_attention_boost_max_extra_focus_budget: int
    vision_attention_boost_min_radius_scale: float
    vision_attention_boost_edge_gain: float
    vision_attention_boost_gaze_sigma_scale: float
    vision_dynamic_track_window: int
    vision_dynamic_candidate_limit_background: int
    vision_dynamic_candidate_limit_focus: int
    vision_dynamic_track_limit: int
    vision_dynamic_summary_limit: int
    vision_dynamic_match_threshold: float
    vision_dynamic_track_forget_ticks: int
    hearing_window_budget: int
    hearing_window_ms: int
    hearing_focus_band_count: int
    hearing_focus_bandwidth_octaves: float
    hearing_attention_boost_enabled: bool
    hearing_attention_boost_decay: float
    hearing_attention_boost_max_extra_window_budget: int
    hearing_attention_boost_max_extra_focus_budget: int
    hearing_attention_boost_min_bandwidth_scale: float
    hearing_attention_boost_focus_gain: float
    hearing_static_dedup_delta_threshold: float
    hearing_static_dedup_band_similarity_threshold: float
    hearing_static_dedup_max_suppression: float
    hearing_auditory_fatigue_decay: float
    hearing_auditory_fatigue_step: float
    hearing_auditory_fatigue_max: float
    r_state_head_limit: int
    r_state_items_per_head: int
    executor_enabled: bool
    executor_dry_run: bool
    executor_max_actions_per_tick: int
    executor_screenshot_enabled: bool
    executor_screenshot_scale: float
    executor_type_interval_ms: int
    autonomous_capture_required: bool
    autonomous_auto_feedback_enabled: bool
    autonomous_idle_backoff_ms: int
    autonomous_stop_on_consecutive_capture_failures: int
    autonomous_stop_on_consecutive_action_errors: int
    autonomous_stop_on_consecutive_idle_ticks: int
    autonomous_teacher_enabled: bool
    autonomous_teacher_mode: str
    autonomous_llm_gate_enabled: bool
    autonomous_llm_gate_mode: str
    autonomous_llm_gate_fail_open: bool
    autonomous_teacher_reward_scale: float
    autonomous_teacher_punishment_scale: float
    autonomous_teacher_repeat_window: int
    autonomous_teacher_repeat_penalty: float
    autonomous_teacher_risky_action_min_drive: float
    autonomous_external_teacher_enabled: bool
    autonomous_external_teacher_mode: str
    autonomous_external_teacher_stub_response_path: str
    autonomous_external_teacher_fail_open: bool
    autonomous_external_teacher_timeout_ms: int
    autonomous_external_teacher_max_retries: int
    autonomous_external_teacher_retry_backoff_ms: int
    autonomous_external_teacher_http_endpoint: str
    autonomous_external_teacher_http_headers: dict[str, Any]
    intrinsic_feedback_enabled: bool
    intrinsic_correctness_reward_gain: float
    intrinsic_dissonance_punishment_gain: float
    intrinsic_surprise_punishment_gain: float
    intrinsic_expectation_tonic_reward_gain: float
    intrinsic_pressure_tonic_punishment_gain: float
    intrinsic_feedback_max_reward_per_tick: float
    intrinsic_feedback_max_punishment_per_tick: float
    cognitive_feeling_habituation_enabled: bool
    cognitive_feeling_habituation_decay: float
    cognitive_feeling_habituation_same_signature_gain: float
    cognitive_feeling_habituation_cross_signature_gain: float
    cognitive_feeling_habituation_signature_change_retention: float
    cognitive_feeling_habituation_surprise_gain: float
    cognitive_feeling_habituation_dissonance_gain: float
    cognitive_feeling_habituation_release_on_grasp_gain: float
    cognitive_feeling_habituation_min_multiplier: float
    time_feeling_enabled: bool
    time_feeling_threshold: float
    time_feeling_gain: float
    time_feeling_min_confidence: float
    time_feeling_default_radius_ticks: float
    time_feeling_recall_gain: float
    time_feeling_fatigue_decay: float
    time_feeling_fatigue_step: float
    time_feeling_fatigue_gain: float
    time_feeling_fatigue_max: float
    motion_feeling_enabled: bool
    motion_feeling_threshold: float
    motion_feeling_gain: float
    motion_feeling_min_confidence: float
    motion_feeling_recall_gain: float
    motion_feeling_attention_gain: float
    motion_feeling_fatigue_decay: float
    motion_feeling_fatigue_step: float
    motion_feeling_fatigue_gain: float
    motion_feeling_fatigue_max: float
    feedback_signal_feeling_enabled: bool
    feedback_signal_feeling_threshold: float
    feedback_signal_feeling_gain: float
    feedback_signal_feeling_min_confidence: float
    feedback_signal_recall_gain: float
    feedback_signal_fatigue_decay: float
    feedback_signal_fatigue_step: float
    feedback_signal_fatigue_gain: float
    feedback_signal_fatigue_max: float
    rhythm_feeling_enabled: bool
    rhythm_window_ticks: int
    rhythm_min_hits: int
    rhythm_min_period_ticks: float
    rhythm_max_period_ticks: float
    rhythm_period_sigma_ratio: float
    rhythm_phase_sigma_ratio: float
    rhythm_recovery_center_ticks: float
    rhythm_recovery_sigma_ticks: float
    rhythm_min_confidence: float
    rhythm_recall_gain: float
    rhythm_pulse_threshold: float
    rhythm_pulse_gain: float
    rhythm_phase_threshold: float
    rhythm_phase_gain: float
    rhythm_fatigue_decay: float
    rhythm_fatigue_step: float
    rhythm_fatigue_gain: float
    rhythm_fatigue_max: float
    vision_auto_surprise_reorient_enabled: bool
    observatory_tick_preview_limit: int
    observatory_tick_list_limit: int
    outputs_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    return _repo_root()


def default_config_path(repo_root_value: Path | None = None) -> Path:
    root = (repo_root_value or repo_root()).resolve()
    return root / "config" / "runtime_config.json"


def load_config(config_path: Path | None = None, *, overrides: dict[str, Any] | None = None) -> AppConfig:
    path = config_path or default_config_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if overrides:
        payload.update(overrides)
    validate_or_raise(payload, load_schema("app_config.schema.json"), label="app_config")
    return AppConfig(**payload)
