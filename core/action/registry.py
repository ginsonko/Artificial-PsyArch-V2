from __future__ import annotations

"""
Action and actuator registry for APV2.1.

The registry is intentionally data-only. It gives the innate rule layer and
DriveManager a shared conflict vocabulary without making either module own the
other's logic.
"""


ACTUATOR_REGISTRY: dict[str, dict] = {
    "actuator::attention_allocation": {
        "label": "attention allocation",
        "external": False,
        "default_per_tick": 1,
        "threshold_range": (0.38, 0.58),
        "conflict_domain": "attention_focus_width_and_anchor",
    },
    "actuator::visual_gaze_center": {
        "label": "visual gaze center",
        "external": False,
        "semi_external": True,
        "default_per_tick": 1,
        "threshold_range": (0.48, 0.72),
        "conflict_domain": "single_visual_center",
    },
    "actuator::visual_focus_scale": {
        "label": "visual focus scale",
        "external": False,
        "semi_external": True,
        "default_per_tick": 1,
        "threshold_range": (0.58, 0.78),
        "conflict_domain": "visual_sampling_scale",
    },
    "actuator::auditory_band_center": {
        "label": "auditory band center",
        "external": False,
        "semi_external": True,
        "default_per_tick": 1,
        "threshold_range": (0.50, 0.70),
        "conflict_domain": "single_auditory_band_center",
    },
    "actuator::auditory_band_width": {
        "label": "auditory band width",
        "external": False,
        "semi_external": True,
        "default_per_tick": 1,
        "threshold_range": (0.48, 0.68),
        "conflict_domain": "auditory_sampling_width",
    },
    "actuator::memory_recall": {
        "label": "memory recall",
        "external": False,
        "default_per_tick": 1,
        "threshold_range": (0.50, 0.78),
        "conflict_domain": "primary_recall_query",
    },
    "actuator::text_editor": {
        "label": "text editor",
        "external": True,
        "default_per_tick": 1,
        "threshold_range": (0.45, 1.05),
        "conflict_domain": "single_text_buffer_edit",
    },
    "actuator::computer_pointer": {
        "label": "computer pointer",
        "external": True,
        "default_per_tick": 1,
        "threshold_range": (0.95, 1.35),
        "conflict_domain": "single_os_pointer",
    },
    "actuator::computer_keyboard": {
        "label": "computer keyboard",
        "external": True,
        "default_per_tick": 1,
        "threshold_range": (1.05, 1.35),
        "conflict_domain": "single_os_keyboard",
    },
    "actuator::llm_call": {
        "label": "llm call",
        "external": True,
        "default_per_tick": 1,
        "threshold_range": (1.05, 1.40),
        "conflict_domain": "single_llm_request",
    },
    "actuator::tool_api": {
        "label": "tool api",
        "external": True,
        "default_per_tick": 1,
        "threshold_range": (1.10, 1.50),
        "conflict_domain": "tool_mutex",
    },
    "actuator::timing": {
        "label": "timing",
        "external": False,
        "default_per_tick": 1,
        "threshold_range": (0.22, 0.45),
        "conflict_domain": "wait_hold_pause",
    },
    "actuator::protective_orientation": {
        "label": "protective orientation",
        "external": False,
        "default_per_tick": 1,
        "threshold_range": (0.42, 0.76),
        "conflict_domain": "protective_orientation",
    },
    # Compatibility-only actuator for the current transitional planner action.
    "actuator::legacy_internal": {
        "label": "legacy internal",
        "external": False,
        "default_per_tick": 1,
        "threshold_range": (0.30, 0.55),
        "conflict_domain": "legacy_internal_prediction",
        "legacy": True,
    },
}


ACTION_NODE_REGISTRY: dict[str, dict] = {
    "action::focus_anchor": {
        "actuator_id": "actuator::attention_allocation",
        "params": ("anchor_label", "strength"),
        "base_threshold": 0.45,
        "fatigue_type": "action_internal",
    },
    "action::continue_focus": {
        "actuator_id": "actuator::attention_allocation",
        "legacy_actuator_id": "actuator::attention",
        "params": ("anchor", "window"),
        "base_threshold": 0.38,
        "fatigue_type": "action_internal",
    },
    "action::diverge_attention": {
        "actuator_id": "actuator::attention_allocation",
        "params": ("family", "top_n_scale"),
        "base_threshold": 0.52,
        "fatigue_type": "action_internal",
    },
    "action::inspect_residual": {
        "actuator_id": "actuator::attention_allocation",
        "legacy_actuator_id": "actuator::attention",
        "params": ("residual_labels",),
        "base_threshold": 0.50,
        "fatigue_type": "mismatch",
    },
    "action::release_focus": {
        "actuator_id": "actuator::attention_allocation",
        "params": ("anchor",),
        "base_threshold": 0.48,
        "fatigue_type": "action_internal",
    },
    "action::move_gaze_to": {
        "actuator_id": "actuator::visual_gaze_center",
        "params": ("x", "y", "target"),
        "base_threshold": 0.68,
        "fatigue_type": "action_internal",
    },
    "action::nudge_gaze": {
        "actuator_id": "actuator::visual_gaze_center",
        "params": ("dx", "dy"),
        "base_threshold": 0.48,
        "fatigue_type": "fast_sensory",
    },
    "action::scan_visual_field": {
        "actuator_id": "actuator::visual_gaze_center",
        "params": ("pattern",),
        "base_threshold": 0.62,
        "fatigue_type": "action_internal",
    },
    "action::hold_gaze": {
        "actuator_id": "actuator::visual_gaze_center",
        "params": ("target",),
        "base_threshold": 0.40,
        "fatigue_type": "positive_validation",
    },
    "action::zoom_visual_focus": {
        "actuator_id": "actuator::visual_focus_scale",
        "params": ("scale", "target"),
        "base_threshold": 0.72,
        "fatigue_type": "action_internal",
    },
    "action::widen_visual_focus": {
        "actuator_id": "actuator::visual_focus_scale",
        "params": ("scale",),
        "base_threshold": 0.62,
        "fatigue_type": "action_internal",
    },
    "action::slide_audio_band": {
        "actuator_id": "actuator::auditory_band_center",
        "params": ("center_hz", "target"),
        "base_threshold": 0.60,
        "fatigue_type": "fast_sensory",
    },
    "action::lock_audio_band": {
        "actuator_id": "actuator::auditory_band_center",
        "params": ("center_hz",),
        "base_threshold": 0.50,
        "fatigue_type": "positive_validation",
    },
    "action::narrow_audio_band": {
        "actuator_id": "actuator::auditory_band_width",
        "params": ("width_hz",),
        "base_threshold": 0.55,
        "fatigue_type": "action_internal",
    },
    "action::widen_audio_band": {
        "actuator_id": "actuator::auditory_band_width",
        "params": ("width_hz",),
        "base_threshold": 0.50,
        "fatigue_type": "action_internal",
    },
    "action::recall_recent_context": {
        "actuator_id": "actuator::memory_recall",
        "params": ("horizon",),
        "base_threshold": 0.55,
        "fatigue_type": "action_internal",
    },
    "action::replay_recent_context": {
        "actuator_id": "actuator::memory_recall",
        "legacy_actuator_id": "actuator::memory",
        "canonical_action_id": "action::recall_recent_context",
        "params": ("horizon",),
        "base_threshold": 0.55,
        "fatigue_type": "action_internal",
        "legacy": True,
    },
    "action::recall_by_timefelt": {
        "actuator_id": "actuator::memory_recall",
        "params": ("delta_t",),
        "base_threshold": 0.62,
        "fatigue_type": "rhythm",
    },
    "action::recall_by_expectation": {
        "actuator_id": "actuator::memory_recall",
        "params": ("b_anchor",),
        "base_threshold": 0.65,
        "fatigue_type": "expectation",
    },
    "action::replay_episode": {
        "actuator_id": "actuator::memory_recall",
        "params": ("episode_id",),
        "base_threshold": 0.76,
        "fatigue_type": "action_internal",
    },
    "action::text_reread": {
        "actuator_id": "actuator::text_editor",
        "params": ("cursor", "span"),
        "base_threshold": 0.45,
        "fatigue_type": "action_external",
        # APV2.1's current text editor mutates only the internal draft buffer.
        # Real-world typing/submission should use OS/commit actuators and stay
        # behind SafetyGate; reread itself is safe internal self-observation.
        "external": False,
    },
    "action::text_insert": {
        "actuator_id": "actuator::text_editor",
        "params": ("token", "text", "cursor"),
        "base_threshold": 0.78,
        "fatigue_type": "action_external",
        # Internal draft edit. It is remembered and learned from, but it is not
        # a real keyboard/OS side effect in this prototype runtime.
        "external": False,
    },
    "action::text_delete": {
        "actuator_id": "actuator::text_editor",
        "params": ("span",),
        "base_threshold": 0.86,
        "fatigue_type": "action_external",
        "external": False,
    },
    "action::text_replace": {
        "actuator_id": "actuator::text_editor",
        "params": ("span", "new_text"),
        "base_threshold": 0.92,
        "fatigue_type": "action_external",
        "external": False,
    },
    "action::text_commit": {
        "actuator_id": "actuator::text_editor",
        "params": ("target_channel",),
        "base_threshold": 1.05,
        "fatigue_type": "action_external",
        # Commit is the boundary where an internal draft may become externally
        # observable, so SafetyGate must continue to review it.
        "external": True,
    },
    "action::pointer_move": {
        "actuator_id": "actuator::computer_pointer",
        "params": ("x", "y"),
        "base_threshold": 0.95,
        "fatigue_type": "action_external",
    },
    "action::pointer_click": {
        "actuator_id": "actuator::computer_pointer",
        "params": ("button", "target"),
        "base_threshold": 1.20,
        "fatigue_type": "action_external",
    },
    "action::pointer_drag": {
        "actuator_id": "actuator::computer_pointer",
        "params": ("from", "to"),
        "base_threshold": 1.32,
        "fatigue_type": "action_external",
    },
    "action::pointer_scroll": {
        "actuator_id": "actuator::computer_pointer",
        "params": ("dy",),
        "base_threshold": 1.05,
        "fatigue_type": "action_external",
    },
    "action::keyboard_type": {
        "actuator_id": "actuator::computer_keyboard",
        "params": ("text",),
        "base_threshold": 1.18,
        "fatigue_type": "action_external",
    },
    "action::keyboard_hotkey": {
        "actuator_id": "actuator::computer_keyboard",
        "params": ("keys",),
        "base_threshold": 1.30,
        "fatigue_type": "action_external",
    },
    "action::llm_think": {
        "actuator_id": "actuator::llm_call",
        "params": ("prompt_context",),
        "base_threshold": 1.10,
        "fatigue_type": "action_external",
    },
    "action::llm_critique": {
        "actuator_id": "actuator::llm_call",
        "params": ("candidate_action",),
        "base_threshold": 1.25,
        "fatigue_type": "action_external",
    },
    "action::llm_write_draft": {
        "actuator_id": "actuator::llm_call",
        "params": ("focus", "context"),
        "base_threshold": 1.35,
        "fatigue_type": "action_external",
    },
    "action::tool_call": {
        "actuator_id": "actuator::tool_api",
        "params": ("tool_name", "args"),
        "base_threshold": 1.25,
        "fatigue_type": "action_external",
    },
    "action::wait": {
        "actuator_id": "actuator::timing",
        "legacy_actuator_id": "actuator::timing",
        "params": ("duration_ticks",),
        "base_threshold": 0.25,
        "fatigue_type": "action_internal",
    },
    "action::avoid": {
        "actuator_id": "actuator::protective_orientation",
        "params": ("target", "reason"),
        "base_threshold": 0.58,
        "fatigue_type": "protective",
        # Generic internal tendency: a later embodied/OS layer may map it to a
        # concrete movement, but APV2.1 should first learn it as "I want to avoid".
        "external": False,
    },
    "action::withdraw": {
        "actuator_id": "actuator::protective_orientation",
        "params": ("target", "reason"),
        "base_threshold": 0.62,
        "fatigue_type": "protective",
        "external": False,
    },
    "action::stabilize_prediction": {
        "actuator_id": "actuator::legacy_internal",
        "legacy_actuator_id": "actuator::prediction",
        "params": ("labels",),
        "base_threshold": 0.32,
        "fatigue_type": "action_internal",
        "legacy": True,
    },
}


def action_meta(action_id: str) -> dict:
    return dict(ACTION_NODE_REGISTRY.get(str(action_id or ""), {}))


def actuator_meta(actuator_id: str) -> dict:
    return dict(ACTUATOR_REGISTRY.get(str(actuator_id or ""), {}))


def action_actuator_id(action_id: str, fallback: str = "actuator::legacy_internal") -> str:
    meta = action_meta(action_id)
    return str(meta.get("actuator_id", "") or fallback)


def is_external_action(action_id: str, actuator_id: str | None = None) -> bool:
    meta = action_meta(action_id)
    resolved_actuator = str(actuator_id or meta.get("actuator_id", "") or "")
    actuator = actuator_meta(resolved_actuator)
    if "external" in meta:
        return bool(meta.get("external", False))
    return bool(actuator.get("external", False))
