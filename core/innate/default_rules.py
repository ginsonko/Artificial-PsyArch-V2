from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleDef:
    rule_id: str
    phase: str
    description: str
    condition: str
    outputs: tuple[dict, ...]
    fatigue_type: str
    threshold: float = 0.0
    anchor: str = "global"


FATIGUE_TYPES: dict[str, dict] = {
    "fast_sensory": {"decay": 0.55, "increase": 0.28, "gain": 0.60, "min_scale": 0.25},
    "mismatch": {"decay": 0.72, "increase": 0.22, "gain": 0.55, "min_scale": 0.18},
    "positive_validation": {"decay": 0.82, "increase": 0.14, "gain": 0.35, "min_scale": 0.45},
    "expectation": {"decay": 0.88, "increase": 0.16, "gain": 0.42, "min_scale": 0.30},
    "action_internal": {"decay": 0.76, "increase": 0.24, "gain": 0.50, "min_scale": 0.22},
    "action_external": {"decay": 0.90, "increase": 0.30, "gain": 0.70, "min_scale": 0.10},
    "rhythm": {"decay": 0.62, "increase": 0.20, "gain": 0.45, "min_scale": 0.28},
    "feedback_learning": {"decay": 0.84, "increase": 0.18, "gain": 0.38, "min_scale": 0.35},
    "safety_gate": {"decay": 0.78, "increase": 0.26, "gain": 0.48, "min_scale": 0.20},
}


def default_rules() -> list[RuleDef]:
    rules: list[RuleDef] = []

    def add(
        rule_id: str,
        phase: str,
        description: str,
        condition: str,
        outputs: tuple[dict, ...],
        fatigue_type: str,
        threshold: float,
        anchor: str = "global",
    ) -> None:
        rules.append(
            RuleDef(
                rule_id=rule_id,
                phase=phase,
                description=description,
                condition=condition,
                outputs=outputs,
                fatigue_type=fatigue_type,
                threshold=threshold,
                anchor=anchor,
            )
        )

    # Cognitive feelings and attention coupling.
    add(
        "CF-001",
        "post_prediction_validation",
        "unexpected real evidence emits surprise and attention bias",
        "positive_pressure",
        ({"type": "emit_sa", "label": "feeling::surprise", "family": "cognitive_feeling"}, {"type": "attention_bias", "bias": "surprise_anchor"}),
        "fast_sensory",
        0.16,
        "top_positive_pressure",
    )
    add(
        "CF-002",
        "post_prediction_validation",
        "over-prediction emits dissonance",
        "negative_pressure",
        ({"type": "emit_sa", "label": "feeling::dissonance", "family": "cognitive_feeling"}, {"type": "attention_bias", "bias": "mismatch_pair"}),
        "mismatch",
        0.16,
        "top_negative_pressure",
    )
    add(
        "CF-003",
        "post_prediction_validation",
        "prediction alignment emits correctness",
        "alignment",
        ({"type": "emit_sa", "label": "feeling::correctness", "family": "cognitive_feeling"},),
        "positive_validation",
        0.18,
    )
    add(
        "CF-004",
        "post_fast_recall",
        "high Bn match efficiency emits grasp",
        "grasp",
        ({"type": "emit_sa", "label": "feeling::grasp", "family": "cognitive_feeling"},),
        "positive_validation",
        0.20,
    )
    add(
        "CF-005",
        "post_prediction_validation",
        "low mismatch and coherent prediction emits coherence",
        "coherence",
        ({"type": "emit_sa", "label": "feeling::coherence", "family": "cognitive_feeling"},),
        "positive_validation",
        0.18,
    )
    add(
        "CF-006",
        "post_slow_recall",
        "multi-peak B/C competition emits uncertainty pressure",
        "uncertainty",
        ({"type": "emit_sa", "label": "feeling::uncertainty", "family": "cognitive_feeling"},),
        "mismatch",
        0.22,
    )
    add(
        "CF-007",
        "emotion_post",
        "reward prediction creates expectation anchor",
        "expectation",
        ({"type": "emit_sa", "label": "feeling::expectation", "family": "cognitive_feeling"},),
        "expectation",
        0.15,
    )
    add(
        "CF-008",
        "emotion_post",
        "punishment prediction creates pressure anchor",
        "pressure",
        ({"type": "emit_sa", "label": "feeling::pressure", "family": "cognitive_feeling"},),
        "expectation",
        0.15,
    )
    add(
        "CF-009",
        "emotion_post",
        "expectation validation emits satisfaction",
        "satisfaction",
        ({"type": "emit_sa", "label": "feeling::satisfaction", "family": "cognitive_feeling"},),
        "expectation",
        0.12,
    )
    add(
        "CF-010",
        "emotion_post",
        "expectation gap emits gap feeling",
        "expectation_gap",
        ({"type": "emit_sa", "label": "feeling::expectation_gap", "family": "cognitive_feeling"},),
        "expectation",
        0.12,
    )
    add(
        "CF-011",
        "post_fast_recall",
        "concentrated B time interval emits time feeling",
        "timefelt",
        ({"type": "trace_log", "topic": "timefelt_from_recall_peak"},),
        "rhythm",
        0.18,
    )
    add(
        "CF-012",
        "post_attention",
        "rhythm phase expectation becomes a felt rhythm anchor",
        "rhythm_phase",
        ({"type": "trace_log", "topic": "rhythm_phase_expectation"},),
        "rhythm",
        0.16,
    )
    add(
        "CF-013",
        "tick_end",
        "runtime overload emits complexity feeling",
        "complexity",
        ({"type": "emit_sa", "label": "runtimefelt::complexity", "family": "runtime_feeling"},),
        "action_internal",
        0.18,
    )
    add(
        "CF-014",
        "tick_end",
        "low novelty and low mismatch emits simplicity feeling",
        "simplicity",
        ({"type": "emit_sa", "label": "runtimefelt::simplicity", "family": "runtime_feeling"},),
        "action_internal",
        0.14,
    )
    add(
        "CF-015",
        "tick_end",
        "high rule or anchor fatigue becomes cognizable",
        "fatigue",
        ({"type": "emit_sa", "label": "feeling::fatigue", "family": "cognitive_feeling"},),
        "action_internal",
        0.28,
    )

    # Attention and action triggers.
    add(
        "AT-001",
        "action_preselect",
        "surprise pulls attention toward new objects",
        "surprise",
        ({"type": "action_node", "action_id": "action::focus_anchor", "drive": 0.18},),
        "fast_sensory",
        0.22,
        "top_positive_pressure",
    )
    add(
        "AT-002",
        "action_preselect",
        "dissonance binds predicted and actual mismatch",
        "dissonance",
        ({"type": "action_node", "action_id": "action::inspect_residual", "drive": 0.20},),
        "mismatch",
        0.20,
    )
    add(
        "AT-003",
        "action_preselect",
        "high grasp and successor clarity keep focus",
        "continue_focus",
        ({"type": "action_node", "action_id": "action::continue_focus", "drive": 0.16},),
        "positive_validation",
        0.18,
    )
    add(
        "AT-004",
        "action_preselect",
        "pressure and residual mass trigger inspection",
        "inspect_residual",
        ({"type": "action_node", "action_id": "action::inspect_residual", "drive": 0.22},),
        "mismatch",
        0.20,
    )
    add(
        "AT-005",
        "action_preselect",
        "coherence lowers divergence drive",
        "coherence",
        ({"type": "action_bias", "action_id": "action::diverge_attention", "drive": -0.12},),
        "positive_validation",
        0.18,
    )
    add(
        "AT-006",
        "action_preselect",
        "novelty and low grasp trigger divergence",
        "novelty",
        ({"type": "action_node", "action_id": "action::diverge_attention", "drive": 0.18},),
        "fast_sensory",
        0.18,
    )
    add(
        "AT-007",
        "action_preselect",
        "high fatigue releases repeated focus",
        "fatigue",
        ({"type": "action_node", "action_id": "action::release_focus", "drive": 0.16},),
        "action_internal",
        0.28,
    )
    add(
        "AT-008",
        "action_preselect",
        "time feeling pulls recall by elapsed interval",
        "timefelt",
        ({"type": "action_node", "action_id": "action::recall_by_timefelt", "drive": 0.18},),
        "rhythm",
        0.18,
    )
    add(
        "AT-009",
        "action_preselect",
        "rhythm phase can prefer waiting",
        "rhythm_phase",
        ({"type": "action_node", "action_id": "action::wait", "drive": 0.12},),
        "rhythm",
        0.16,
    )

    # Learning interfaces.
    add(
        "BC-001",
        "tick_end",
        "positive cognitive pressure emits online positive-pair learning event",
        "positive_pressure",
        ({"type": "learning_event", "event": "positive_pair"},),
        "mismatch",
        0.18,
    )
    add(
        "BC-002",
        "tick_end",
        "negative cognitive pressure emits online negative-pair learning event",
        "negative_pressure",
        ({"type": "learning_event", "event": "negative_pair"},),
        "mismatch",
        0.18,
    )
    add(
        "BC-003",
        "tick_end",
        "ordered text/audio/vision evidence emits transition learning event",
        "transition",
        ({"type": "learning_event", "event": "transition"},),
        "action_internal",
        0.12,
    )
    add(
        "BC-004",
        "tick_end",
        "action feedback emits action outcome learning event",
        "action_feedback",
        ({"type": "learning_event", "event": "action_outcome"},),
        "feedback_learning",
        0.08,
    )
    add(
        "BC-005",
        "tick_end",
        "multimodal co-presence emits binding learning event",
        "multimodal_binding",
        ({"type": "learning_event", "event": "multimodal_binding"},),
        "fast_sensory",
        0.10,
    )
    add(
        "BC-006",
        "emotion_post",
        "expectation anchor schedules future verification",
        "expectation",
        ({"type": "learning_event", "event": "verify_b_anchor"},),
        "expectation",
        0.16,
    )

    # Emotion rules. EmotionModulator applies the actual slow state; these are
    # auditable innate deltas that can be merged with the existing CFS/Rwd/Pun path.
    emotion_rules = (
        ("EM-DA-001", "reward", "DA", 0.10, "positive_validation", "reward increases drive"),
        ("EM-DA-002", "punishment", "DA", -0.08, "mismatch", "punishment lowers drive"),
        ("EM-ADR-001", "surprise", "ADR", 0.09, "fast_sensory", "surprise increases arousal"),
        ("EM-ADR-002", "coherence", "ADR", -0.05, "positive_validation", "coherence lowers arousal"),
        ("EM-OXY-001", "social_reward", "OXY", 0.08, "positive_validation", "positive social feedback increases trust"),
        ("EM-OXY-002", "social_punishment", "OXY", -0.07, "expectation", "social conflict reduces trust"),
        ("EM-SER-001", "correctness", "SER", 0.08, "positive_validation", "correctness increases stability"),
        ("EM-SER-002", "dissonance", "SER", -0.06, "mismatch", "mismatch lowers stability"),
        ("EM-END-001", "relief", "END", 0.08, "expectation", "relief increases recovery"),
        ("EM-END-002", "sustained_pressure", "END", -0.05, "expectation", "unrelieved pressure consumes recovery"),
        ("EM-COR-001", "risk", "COR", 0.10, "mismatch", "risk increases caution"),
        ("EM-COR-002", "safe_validation", "COR", -0.06, "positive_validation", "safe validation lowers caution"),
        ("EM-NOV-001", "novelty", "NOV", 0.09, "fast_sensory", "novelty increases exploration"),
        ("EM-NOV-002", "familiarity", "NOV", -0.05, "action_internal", "familiarity lowers exploration"),
        ("EM-FOC-001", "continue_focus", "FOC", 0.08, "positive_validation", "clear successor increases focus"),
        ("EM-FOC-002", "uncertainty", "FOC", -0.06, "action_internal", "uncertainty lowers focus lock"),
    )
    for rule_id, condition, channel, delta, fatigue_type, description in emotion_rules:
        add(
            rule_id,
            "emotion_post",
            description,
            condition,
            ({"type": "emotion_delta", "channel": channel, "delta": delta},),
            fatigue_type,
            0.12,
        )

    # Action triggers and safety scaffolding.
    action_rules = (
        ("AC-001", "visual_surprise", "action::move_gaze_to", 0.18, "fast_sensory", "visual surprise moves gaze"),
        ("AC-002", "visual_motion", "action::nudge_gaze", 0.14, "fast_sensory", "high motion triggers tracking"),
        ("AC-003", "novelty", "action::scan_visual_field", 0.12, "action_internal", "novel scenes trigger scanning"),
        ("AC-004", "visual_uncertainty", "action::zoom_visual_focus", 0.13, "action_internal", "small uncertain object triggers zoom"),
        ("AC-005", "audio_surprise", "action::slide_audio_band", 0.16, "fast_sensory", "audio onset moves auditory focus"),
        ("AC-006", "voice_like", "action::lock_audio_band", 0.14, "positive_validation", "voice-like audio locks band"),
        ("AC-007", "audio_low_grasp", "action::widen_audio_band", 0.12, "action_internal", "unknown audio widens band"),
        ("AC-008", "text_mismatch", "action::text_reread", 0.14, "mismatch", "text mismatch triggers reread"),
        ("AC-009", "expected_token", "action::text_insert", 0.12, "action_external", "strong expected token prepares insert"),
        ("AC-010", "text_revision", "action::text_replace", 0.15, "action_external", "mismatch with target triggers replace"),
        ("AC-011", "text_commit_ready", "action::text_commit", 0.14, "action_external", "correct low-pressure draft can commit"),
        ("AC-012", "expectation", "action::recall_by_expectation", 0.14, "expectation", "expectation anchor triggers recall"),
        ("AC-013", "pressure_external_candidate", "action::replay_episode", 0.14, "expectation", "pressure before action triggers replay"),
        ("AC-014", "timefelt", "action::recall_by_timefelt", 0.16, "rhythm", "time feeling triggers temporal recall"),
        ("AC-015", "uncertainty", "action::wait", 0.12, "action_internal", "multi-peak uncertainty triggers wait"),
        ("AC-017", "ui_goal", "action::pointer_move", 0.10, "action_external", "UI target prepares pointer move"),
        ("AC-018", "click_ready", "action::pointer_click", 0.10, "action_external", "safe target prepares click"),
        ("AC-019", "hard_task", "action::llm_think", 0.10, "action_external", "hard task can call LLM actuator"),
        ("AC-020", "external_risk", "action::llm_critique", 0.10, "action_external", "external risk can call critique"),
    )
    for rule_id, condition, action_id, drive, fatigue_type, description in action_rules:
        add(
            rule_id,
            "action_preselect",
            description,
            condition,
            ({"type": "action_node", "action_id": action_id, "drive": drive},),
            fatigue_type,
            0.16,
        )
    add(
        "AC-016",
        "action_preselect",
        "high external risk triggers safety gate instead of competing as an action",
        "external_risk",
        ({"type": "safety_gate", "decision": "inhibit_external"},),
        "safety_gate",
        0.18,
    )

    # Action feedback family.
    for rule_id, condition, event, fatigue_type, threshold in (
        ("AF-001", "action_selected", "action_selection", "feedback_learning", 0.05),
        ("AF-002", "action_selected", "causal_window", "feedback_learning", 0.05),
        ("AF-003", "action_feedback", "action_feedback", "feedback_learning", 0.05),
        ("AF-004", "positive_action_feedback", "reward_signal", "positive_validation", 0.05),
        ("AF-005", "negative_action_feedback", "punishment_signal", "feedback_learning", 0.05),
        ("AF-006", "positive_action_feedback", "action_outcome_success", "positive_validation", 0.05),
        ("AF-007", "negative_action_feedback", "action_outcome_failure", "feedback_learning", 0.05),
        ("AF-008", "action_prediction_error", "action_prediction_error", "mismatch", 0.05),
        ("AF-009", "successor_action_feedback", "action_consequence_estimate", "action_internal", 0.05),
        ("AF-010", "memory_predicted_action", "memory_predicted_action", "feedback_learning", 0.05),
        ("AF-011", "action_inhibition", "action_inhibition", "safety_gate", 0.05),
    ):
        add(
            rule_id,
            "tick_end" if rule_id not in {"AF-009", "AF-010", "AF-011"} else "action_preselect",
            f"{event} learning/interface trace",
            condition,
            ({"type": "learning_event", "event": event},),
            fatigue_type,
            threshold,
        )

    return rules

