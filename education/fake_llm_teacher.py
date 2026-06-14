from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from education.intervention import normalize_education_intervention


def _round4(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return round(float(default), 4)


@dataclass
class FakeLLMBuildingBlockTeacher:
    """Deterministic LLM-shaped teacher for generalization tests.

    The class intentionally mimics the output contract of an LLM teacher while
    staying deterministic and free. It never supplies the target sentence as a
    text-insert parameter; it only hints process actions and rewards a finished
    combination after AP has produced it.
    """

    taught_blocks: list[dict]
    goal: str = "combine taught blocks into a reasonable unseen phrase"
    source: str = "fake_llm_teacher"

    def propose_intervention(
        self,
        *,
        tick_index: int,
        draft_context: dict,
        expected_text: dict,
        composed_text: str,
        combination_complete: bool,
        final_reread_seen: bool,
    ) -> dict:
        visible_length = int((draft_context or {}).get("visible_length", 0) or 0)
        visible_text = str((draft_context or {}).get("visible_text", "") or "")
        last_event_type = str((draft_context or {}).get("last_event_type", "") or "")
        last_insert_age = int((draft_context or {}).get("last_insert_age", 9999) or 9999)
        last_reread_age = int((draft_context or {}).get("last_reread_age", 9999) or 9999)
        expected_token = str((expected_text or {}).get("token", "") or "")

        action_biases: list[dict] = []
        if visible_length > 0 and last_event_type == "insert" and last_insert_age <= 1:
            action_biases.append(
                {
                    "action_id": "action::text_reread",
                    "drive_delta": 0.56,
                    "params": {"span": [0, visible_length], "reason": "fake_llm_teacher_review_after_insert"},
                    "notes": ["fake_llm_process_hint", "review_after_each_building_block_piece"],
                }
            )
        if visible_text and not combination_complete:
            action_biases.append(
                {
                    "action_id": "action::text_commit",
                    "drive_delta": -1.25,
                    "params": {"reason": "fake_llm_teacher_not_complete_yet"},
                    "notes": [
                        "do_not_commit_before_unseen_combination_finishes",
                        "negative_soft_bias_only",
                        "strong_enough_to_prevent_half_phrase_demo_send",
                    ],
                }
            )
        if visible_text and not expected_token and not combination_complete and last_reread_age > 1:
            action_biases.append(
                {
                    "action_id": "action::wait",
                    "drive_delta": 0.14,
                    "params": {"duration_ticks": 1, "reason": "fake_llm_teacher_wait_for_next_block"},
                    "notes": ["wait_when_next_block_not_clear"],
                }
            )
        if combination_complete and final_reread_seen:
            action_biases.append(
                {
                    "action_id": "action::text_commit",
                    "drive_delta": 0.92,
                    "params": {"target_channel": "internal_draft", "reason": "fake_llm_teacher_commit_after_unseen_combo_reread"},
                    "notes": ["commit_after_self_reread", "teacher_does_not_supply_final_text"],
                }
            )

        state_items = [
            {
                "sa_label": "education_hint::building_block_generalization",
                "display_text": "teacher hint: combine learned blocks",
                "family": "education_intervention",
                "source_type": "external_teacher",
                "real_energy": 0.18,
                "cognitive_pressure": 0.08,
                "anchor_meta": {
                    "schema_id": "fake_llm_building_block_hint/v1",
                    "goal": self.goal,
                    "taught_block_ids": [str(block.get("block_id", "") or "") for block in self.taught_blocks],
                    "does_not_provide_final_answer": True,
                    "visible_length": visible_length,
                    "combination_complete": bool(combination_complete),
                },
            }
        ]
        return normalize_education_intervention(
            {
                "source": self.source,
                "teacher_kind": "fake_llm",
                "goal": self.goal,
                "state_items": state_items,
                "action_biases": action_biases,
                "feedback": {},
                "notes": [
                    "fake_llm_same_schema_as_real_llm",
                    "process_hint_only",
                    "no_final_answer_in_action_params",
                ],
            },
            tick_index=tick_index,
        )

    def evaluate_after_step(
        self,
        *,
        committed_text: str,
        composed_text: str,
        taught_examples: list[str],
    ) -> dict:
        if not committed_text:
            return {"schema_id": "fake_llm_teacher_evaluation/v1", "available": False, "feedback": {}}
        unseen = composed_text not in set(taught_examples)
        matched = committed_text == composed_text
        feedback = {
            "reward": 0.0,
            "punishment": 0.0,
            "correctness": 0.0,
            "confidence": 0.86,
            "source": "fake_llm_teacher_generalization_reward",
            "notes": ["post_output_teacher_evaluation"],
        }
        if matched and unseen:
            feedback.update(
                {
                    "reward": 0.62,
                    "correctness": 0.74,
                    "notes": [
                        "post_output_teacher_evaluation",
                        "reasonable_unseen_combination",
                        "reward_after_ap_generated_combo",
                    ],
                }
            )
        elif committed_text:
            feedback.update(
                {
                    "punishment": 0.18,
                    "notes": ["post_output_teacher_evaluation", "combo_not_matched"],
                }
            )
        return {
            "schema_id": "fake_llm_teacher_evaluation/v1",
            "available": True,
            "matched": bool(matched),
            "unseen_combination": bool(unseen),
            "committed_text": committed_text,
            "composed_text": composed_text,
            "feedback": {**feedback, "reward": _round4(feedback["reward"]), "correctness": _round4(feedback["correctness"]), "punishment": _round4(feedback["punishment"])},
            "meaning": "teacher rewards only after AP has produced the combination",
        }


@dataclass
class FakeLLMNoisyGeneralizationTeacher:
    """LLM-shaped teacher for noisy generalization experiments.

    This teacher is still outside AP core. It can keep a childlike writing
    attempt alive, encourage rereading, and evaluate the result after commit,
    but it never writes the final text into the draft. That boundary lets the
    experiment test AP's action loop instead of testing a hidden answer script.
    """

    taught_blocks: list[dict]
    goal: str = "combine taught blocks under noise, near matches, and interference"
    source: str = "fake_llm_noisy_generalization_teacher"

    def propose_intervention(
        self,
        *,
        tick_index: int,
        draft_context: dict,
        expected_text: dict,
        attempt_complete: bool,
        final_reread_seen: bool,
        scene_pressure: float = 0.0,
        task_hint: str = "",
    ) -> dict:
        visible_length = int((draft_context or {}).get("visible_length", 0) or 0)
        visible_text = str((draft_context or {}).get("visible_text", "") or "")
        last_event_type = str((draft_context or {}).get("last_event_type", "") or "")
        last_insert_age = int((draft_context or {}).get("last_insert_age", 9999) or 9999)
        last_reread_age = int((draft_context or {}).get("last_reread_age", 9999) or 9999)
        expected_token = str((expected_text or {}).get("token", "") or "")
        ambiguity = _round4(float((expected_text or {}).get("ambiguity", 0.0) or 0.0))

        action_biases: list[dict] = []
        if visible_length > 0 and last_event_type in {"insert", "replace"} and last_insert_age <= 1:
            action_biases.append(
                {
                    "action_id": "action::text_reread",
                    "drive_delta": 0.54,
                    "params": {"span": [0, visible_length], "reason": "fake_llm_noisy_review_after_insert"},
                    "notes": ["fake_llm_process_hint", "review_current_attempt_after_write"],
                }
            )
        if visible_text and not attempt_complete:
            action_biases.append(
                {
                    "action_id": "action::text_commit",
                    "drive_delta": -1.72,
                    "params": {"reason": "fake_llm_noisy_attempt_not_complete_yet"},
                    "notes": [
                        "do_not_commit_before_current_attempt_finishes",
                        "negative_soft_bias_only",
                        "does_not_evaluate_semantics_before_commit",
                        "stronger_longrun_premature_commit_guard",
                    ],
                }
            )
        if visible_text and not expected_token and not attempt_complete and last_reread_age > 1:
            action_biases.append(
                {
                    "action_id": "action::wait",
                    "drive_delta": 0.16 + min(0.08, ambiguity * 0.12),
                    "params": {"duration_ticks": 1, "reason": "fake_llm_noisy_wait_for_next_block"},
                    "notes": ["wait_when_next_block_is_unclear", f"ambiguity={ambiguity}"],
                }
            )
        if attempt_complete and final_reread_seen:
            action_biases.append(
                {
                    "action_id": "action::text_commit",
                    "drive_delta": 0.88,
                    "params": {"target_channel": "internal_draft", "reason": "fake_llm_noisy_commit_after_reread"},
                    "notes": ["commit_after_self_reread", "teacher_does_not_supply_final_text"],
                }
            )

        state_items = [
            {
                "sa_label": "education_hint::noisy_building_block_generalization",
                "display_text": "teacher hint: try, reread, then learn from feedback",
                "family": "education_intervention",
                "source_type": "external_teacher",
                "real_energy": 0.16 + min(0.08, max(0.0, float(scene_pressure or 0.0)) * 0.24),
                "cognitive_pressure": 0.07 + min(0.10, max(0.0, float(scene_pressure or 0.0)) * 0.18),
                "anchor_meta": {
                    "schema_id": "fake_llm_noisy_generalization_hint/v1",
                    "goal": self.goal,
                    "task_hint": str(task_hint or ""),
                    "taught_block_ids": [str(block.get("block_id", "") or "") for block in self.taught_blocks],
                    "does_not_provide_final_answer": True,
                    "attempt_complete": bool(attempt_complete),
                    "visible_length": visible_length,
                    "expected_candidate_count": int((expected_text or {}).get("candidate_count", 0) or 0),
                    "expected_ambiguity": ambiguity,
                },
            }
        ]
        return normalize_education_intervention(
            {
                "source": self.source,
                "teacher_kind": "fake_llm",
                "goal": self.goal,
                "state_items": state_items,
                "action_biases": action_biases,
                "feedback": {},
                "notes": [
                    "fake_llm_same_schema_as_real_llm",
                    "process_hint_only",
                    "post_output_evaluation_only",
                    "no_final_answer_in_action_params",
                ],
            },
            tick_index=tick_index,
        )

    def evaluate_after_step(
        self,
        *,
        committed_text: str,
        target_text: str,
        taught_examples: list[str],
        near_variants: list[str] | None = None,
        trial_kind: str = "",
    ) -> dict:
        if not committed_text:
            return {"schema_id": "fake_llm_noisy_teacher_evaluation/v1", "available": False, "feedback": {}}
        near_set = {str(row) for row in list(near_variants or []) if str(row)}
        taught_set = {str(row) for row in list(taught_examples or []) if str(row)}
        exact = str(committed_text) == str(target_text)
        near = bool(not exact and str(committed_text) in near_set)
        unseen = str(target_text) not in taught_set and str(committed_text) not in taught_set
        feedback = {
            "reward": 0.0,
            "punishment": 0.0,
            "correctness": 0.0,
            "confidence": 0.84,
            "source": "fake_llm_teacher_noisy_generalization_feedback",
            "notes": ["post_output_teacher_evaluation", f"trial_kind={trial_kind}"],
        }
        if exact and unseen:
            feedback.update(
                {
                    "reward": 0.68,
                    "correctness": 0.80,
                    "notes": [
                        "post_output_teacher_evaluation",
                        "exact_unseen_combination",
                        "reward_after_ap_generated_combo",
                        f"trial_kind={trial_kind}",
                    ],
                }
            )
            grade = "exact"
        elif near and unseen:
            feedback.update(
                {
                    "reward": 0.36,
                    "correctness": 0.52,
                    "punishment": 0.02,
                    "notes": [
                        "post_output_teacher_evaluation",
                        "near_semantic_match",
                        "partial_reward_childlike_flexible_grammar",
                        f"trial_kind={trial_kind}",
                    ],
                }
            )
            grade = "near"
        else:
            feedback.update(
                {
                    "punishment": 0.30,
                    "correctness": 0.08,
                    "notes": [
                        "post_output_teacher_evaluation",
                        "mismatch_or_wrong_binding",
                        "punishment_after_ap_output",
                        f"trial_kind={trial_kind}",
                    ],
                }
            )
            grade = "wrong"
        return {
            "schema_id": "fake_llm_noisy_teacher_evaluation/v1",
            "available": True,
            "grade": grade,
            "exact_match": bool(exact),
            "near_match": bool(near),
            "unseen_combination": bool(unseen),
            "committed_text": str(committed_text),
            "target_text": str(target_text),
            "feedback": {
                **feedback,
                "reward": _round4(feedback["reward"]),
                "correctness": _round4(feedback["correctness"]),
                "punishment": _round4(feedback["punishment"]),
            },
            "meaning": "teacher evaluates after AP commits; exact gets full reward, near gets partial reward, wrong gets punishment",
        }
