from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from education.intervention import normalize_education_intervention


DEFAULT_GUARDRAILS = [
    "external teacher, not AP core",
    "soft bias only; AP may accept, ignore, or outcompete it",
    "all-SA state-field citizenship is preserved",
    "parameterized action hints must keep params visible",
    "teacher-off emits no state_items, action_biases, or feedback",
    "no hardcoded answers in teacher-off probes",
]

DEFAULT_PHASES = [
    {
        "phase_id": "demonstrate",
        "title": "demonstrate",
        "strength": 1.0,
        "teacher_signal": {"state_items": True, "action_biases": True, "feedback": True},
        "notes": ["full external teacher demonstration"],
    },
    {
        "phase_id": "strong_scaffold",
        "title": "strong scaffold",
        "strength": 0.86,
        "teacher_signal": {"state_items": True, "action_biases": True, "feedback": True},
        "notes": ["teacher gives state hints, soft drive bias, and feedback"],
    },
    {
        "phase_id": "weak_scaffold",
        "title": "weak scaffold",
        "strength": 0.38,
        "teacher_signal": {"state_items": True, "action_biases": True, "feedback": True},
        "notes": ["teacher hint is fading; AP competition matters more"],
    },
    {
        "phase_id": "feedback_only",
        "title": "feedback only",
        "strength": 0.72,
        "teacher_signal": {"state_items": False, "action_biases": False, "feedback": True},
        "notes": ["teacher only rewards or punishes outcome evidence"],
    },
    {
        "phase_id": "teacher_off",
        "title": "teacher off",
        "strength": 0.0,
        "teacher_signal": {"state_items": False, "action_biases": False, "feedback": False},
        "notes": ["teacher-off boundary: AP must run without teacher signal"],
    },
    {
        "phase_id": "cold_retest",
        "title": "cold retest",
        "strength": 0.0,
        "teacher_signal": {"state_items": False, "action_biases": False, "feedback": False},
        "notes": ["fresh or cleared context retest without teacher signal"],
    },
]


def _round4(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return round(float(default), 4)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    numeric = _round4(value, default)
    return max(float(low), min(float(high), numeric))


def _clean_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_dict(value: Any) -> dict:
    return dict(value or {}) if isinstance(value, dict) else {}


def _unique_notes(*groups: Any) -> list[str]:
    notes: list[str] = []
    for group in groups:
        for item in _as_list(group):
            text = str(item or "").strip()
            if text and text not in notes:
                notes.append(text)
    return notes


def _normalize_teacher_signal(value: Any) -> dict:
    if isinstance(value, dict):
        raw = dict(value)
        return {
            "state_items": bool(raw.get("state_items", False)),
            "action_biases": bool(raw.get("action_biases", False)),
            "feedback": bool(raw.get("feedback", False)),
        }
    allowed = {str(item) for item in _as_list(value)}
    return {
        "state_items": "state_items" in allowed,
        "action_biases": "action_biases" in allowed,
        "feedback": "feedback" in allowed,
    }


def _normalize_phase(raw_phase: dict) -> dict:
    phase = dict(raw_phase or {})
    phase_id = _clean_id(phase.get("phase_id"), "strong_scaffold")
    return {
        "phase_id": phase_id,
        "title": str(phase.get("title", "") or phase_id),
        "strength": _clamp(phase.get("strength", 0.0), 0.0, 1.0),
        "teacher_signal": _normalize_teacher_signal(phase.get("teacher_signal", {})),
        "notes": _unique_notes(phase.get("notes", [])),
        "success_criteria": _as_list(phase.get("success_criteria", [])),
        "failure_policy": str(phase.get("failure_policy", "") or ""),
    }


def _normalize_building_block(raw_block: dict, index: int) -> dict:
    block = dict(raw_block or {})
    block_id = _clean_id(block.get("block_id") or block.get("id"), f"block::{index}")
    return {
        "block_id": block_id,
        "label": str(block.get("label", "") or block_id),
        "modality": str(block.get("modality", "") or "mixed"),
        "role": str(block.get("role", "") or "building_block"),
        "notes": _unique_notes(block.get("notes", [])),
    }


def _normalize_step(raw_step: dict, index: int) -> dict:
    step = dict(raw_step or {})
    step_id = _clean_id(step.get("step_id") or step.get("id"), f"step::{index}")
    return {
        "step_id": step_id,
        "title": str(step.get("title", "") or step_id),
        "scenario_tags": [str(item) for item in _as_list(step.get("scenario_tags", [])) if str(item or "").strip()],
        "state_items": [dict(item) for item in _as_list(step.get("state_items", [])) if isinstance(item, dict)],
        "action_biases": [dict(item) for item in _as_list(step.get("action_biases", [])) if isinstance(item, dict)],
        "feedback": _as_dict(step.get("feedback", {})),
        "expected_evidence": [str(item) for item in _as_list(step.get("expected_evidence", [])) if str(item or "").strip()],
        "notes": _unique_notes(step.get("notes", [])),
    }


def normalize_skill_scaffold_spec_v2(raw_spec: dict) -> dict:
    """Normalize a teacher-side skill course specification.

    The returned dict is deliberately data-only. Domain details live in this
    spec, while the controller below only applies generic phase gates, strength
    scaling, teacher-off stripping, and mastery annealing.
    """

    spec = dict(raw_spec or {})
    skill_id = _clean_id(spec.get("skill_id"), "skill::unnamed")
    raw_phases = [dict(row) for row in _as_list(spec.get("phases", [])) if isinstance(row, dict)]
    phases = [_normalize_phase(row) for row in (raw_phases or DEFAULT_PHASES)]
    phase_ids = {row["phase_id"] for row in phases}
    for default_phase in DEFAULT_PHASES:
        if default_phase["phase_id"] not in phase_ids:
            phases.append(_normalize_phase(default_phase))
    return {
        "schema_id": "skill_scaffold_spec/v2",
        "skill_id": skill_id,
        "title": str(spec.get("title", "") or skill_id),
        "goal": str(spec.get("goal", "") or ""),
        "domain_tags": [str(item) for item in _as_list(spec.get("domain_tags", [])) if str(item or "").strip()],
        "building_blocks": [
            _normalize_building_block(row, index)
            for index, row in enumerate(_as_list(spec.get("building_blocks", [])))
            if isinstance(row, dict)
        ],
        "phases": phases,
        "steps": [
            _normalize_step(row, index)
            for index, row in enumerate(_as_list(spec.get("steps", [])))
            if isinstance(row, dict)
        ],
        "mastery": {
            "success_threshold": _clamp(_as_dict(spec.get("mastery", {})).get("success_threshold", 0.42), 0.0, 1.0),
            "update_rate": _clamp(_as_dict(spec.get("mastery", {})).get("update_rate", 0.30), 0.01, 1.0, default=0.30),
            "decay_per_mastery": _clamp(
                _as_dict(spec.get("mastery", {})).get("decay_per_mastery", 0.72),
                0.0,
                0.95,
                default=0.72,
            ),
            "warmup_successes": max(1, int(_as_dict(spec.get("mastery", {})).get("warmup_successes", 6) or 6)),
        },
        "guardrails": _unique_notes(spec.get("guardrails", []), DEFAULT_GUARDRAILS),
        "notes": _unique_notes(spec.get("notes", [])),
    }


@dataclass
class SkillProtocolStateV2:
    skill_id: str
    enabled: bool = False
    phase_id: str = "strong_scaffold"
    base_strength: float = 1.0
    mastery_estimate: float = 0.0
    success_steps: int = 0
    failure_steps: int = 0
    last_step_id: str = ""
    last_intervention: dict = field(default_factory=dict)
    last_effective_strength: float = 0.0


class SkillScaffoldProtocolV2Controller:
    """Generic external teacher protocol for many skills.

    It is intentionally outside AP core. It emits ordinary
    ``education_intervention/v1`` packets: state materials, soft action drive
    biases, and reward/punishment feedback. It never executes actions and it
    strips every teacher signal in teacher-off/cold-retest phases.
    """

    def __init__(self, specs: list[dict] | None = None) -> None:
        self._specs: dict[str, dict] = {}
        self._states: dict[str, SkillProtocolStateV2] = {}
        for spec in list(specs or []):
            self.register_spec(spec)

    def register_spec(self, spec: dict) -> dict:
        normalized = normalize_skill_scaffold_spec_v2(spec)
        skill_id = normalized["skill_id"]
        self._specs[skill_id] = normalized
        if skill_id not in self._states:
            first_phase = normalized["phases"][0]["phase_id"] if normalized["phases"] else "strong_scaffold"
            self._states[skill_id] = SkillProtocolStateV2(skill_id=skill_id, phase_id=first_phase)
        return self.phase_summary(skill_id, reason="register_spec")

    def enable(self, skill_id: str, *, phase_id: str | None = None, strength: float | None = None) -> dict:
        state = self._state(skill_id)
        state.enabled = True
        if phase_id is not None:
            self.set_phase(skill_id, phase_id)
        if strength is not None:
            state.base_strength = _clamp(strength, 0.0, 1.0)
        return self.phase_summary(skill_id, reason="teacher_enable")

    def disable(self, skill_id: str) -> dict:
        state = self._state(skill_id)
        state.enabled = False
        state.last_intervention = {}
        state.last_step_id = ""
        state.last_effective_strength = 0.0
        return self.phase_summary(skill_id, reason="teacher_disable")

    def set_phase(self, skill_id: str, phase_id: str) -> dict:
        state = self._state(skill_id)
        phase = self._phase(skill_id, phase_id)
        state.phase_id = phase["phase_id"]
        return self.phase_summary(skill_id, reason="set_phase")

    def build_intervention(
        self,
        *,
        skill_id: str,
        step_id: str,
        tick_index: int,
        context: dict | None = None,
        observed: dict | None = None,
    ) -> dict:
        spec = self._spec(skill_id)
        state = self._state(skill_id)
        phase = self._phase(skill_id, state.phase_id)
        step = self._step(skill_id, step_id)
        signal = dict(phase.get("teacher_signal", {}) or {})
        effective = self.effective_strength(skill_id)
        context = dict(context or {})
        observed = dict(observed or {})

        if not state.enabled:
            signal = {"state_items": False, "action_biases": False, "feedback": False}
            effective = 0.0

        state_items = self._scaled_state_items(
            step.get("state_items", []),
            skill_id=skill_id,
            step_id=step["step_id"],
            phase_id=phase["phase_id"],
            strength=effective,
        )
        action_biases = self._scaled_action_biases(
            step.get("action_biases", []),
            skill_id=skill_id,
            step_id=step["step_id"],
            phase_id=phase["phase_id"],
            strength=effective,
        )
        feedback = self._scaled_feedback(step.get("feedback", {}), phase_id=phase["phase_id"], strength=effective)

        if not signal.get("state_items", False):
            state_items = []
        if not signal.get("action_biases", False):
            action_biases = []
        if not signal.get("feedback", False):
            feedback = {}

        is_teacher_off = phase["phase_id"] in {"teacher_off", "cold_retest"} or not state.enabled
        notes = _unique_notes(
            [
                "Skill Scaffold Protocol v2",
                "external teacher, not AP core",
                "soft bias only",
                "all-SA state-field materials are allowed",
                "parameterized action hints preserved",
                "no hardcoded answers in teacher-off",
            ],
            spec.get("guardrails", []),
            phase.get("notes", []),
            step.get("notes", []),
            context.get("notes", []),
            observed.get("notes", []),
            ["teacher-off boundary active" if is_teacher_off else "teacher signal may be offered as soft bias"],
        )

        intervention = normalize_education_intervention(
            {
                "schema_id": "education_intervention/v1",
                "source": "skill_scaffold_protocol_v2",
                "teacher_kind": "external_skill_teacher_v2",
                "goal": spec.get("goal", ""),
                "tick_index": int(tick_index),
                "state_items": state_items,
                "action_biases": action_biases,
                "feedback": feedback,
                "notes": notes,
            },
            tick_index=tick_index,
        )
        intervention["skill_protocol_v2"] = {
            "schema_id": "skill_scaffold_intervention_trace/v2",
            "skill_id": skill_id,
            "step_id": step["step_id"],
            "phase_id": phase["phase_id"],
            "enabled": bool(state.enabled),
            "effective_strength": _round4(effective),
            "mastery_estimate": _round4(state.mastery_estimate),
            "teacher_signal": signal,
            "teacher_off_boundary": {
                "active": bool(is_teacher_off),
                "state_item_count": len(intervention.get("state_items", []) or []),
                "action_bias_count": len(intervention.get("action_biases", []) or []),
                "has_feedback": bool(intervention.get("feedback", {})),
            },
            "expected_evidence": list(step.get("expected_evidence", []) or []),
        }
        state.last_step_id = step["step_id"]
        state.last_intervention = intervention
        state.last_effective_strength = _round4(effective)
        return intervention

    def observe_result(
        self,
        *,
        skill_id: str,
        selected_actions: list[dict] | None = None,
        feedback: dict | None = None,
        observed: dict | None = None,
        expected_evidence: list[str] | None = None,
    ) -> dict:
        spec = self._spec(skill_id)
        state = self._state(skill_id)
        selected_actions = [dict(row) for row in list(selected_actions or []) if isinstance(row, dict)]
        feedback = dict(feedback or {})
        observed = dict(observed or {})

        hinted_ids = {
            str(row.get("action_id", "") or "")
            for row in list(state.last_intervention.get("action_biases", []) or [])
            if isinstance(row, dict)
        }
        selected_ids = {str(row.get("action_id", "") or "") for row in selected_actions}
        selected_hint = bool(hinted_ids and selected_ids.intersection(hinted_ids))

        expected = set(expected_evidence or state.last_intervention.get("skill_protocol_v2", {}).get("expected_evidence", []) or [])
        observed_tokens = set()
        for key in ("evidence", "event_types", "selected_action_ids", "state_labels"):
            observed_tokens.update(str(item) for item in _as_list(observed.get(key, [])) if str(item or "").strip())
        evidence_hits = sorted(expected.intersection(observed_tokens))
        evidence_score = len(evidence_hits) / max(1, len(expected)) if expected else 0.0

        reward = _clamp(feedback.get("reward", 0.0), 0.0, 1.0)
        correctness = _clamp(feedback.get("correctness", 0.0), 0.0, 1.0)
        punishment = _clamp(feedback.get("punishment", 0.0), 0.0, 1.0)
        success_score = _round4(
            (0.30 if selected_hint else 0.0)
            + evidence_score * 0.30
            + reward * 0.20
            + correctness * 0.20
            - punishment * 0.35
        )
        threshold = float(spec.get("mastery", {}).get("success_threshold", 0.42) or 0.42)
        succeeded = success_score >= threshold
        if succeeded:
            state.success_steps += 1
        else:
            state.failure_steps += 1

        total = max(1, state.success_steps + state.failure_steps)
        success_rate = state.success_steps / total
        warmup = max(1, int(spec.get("mastery", {}).get("warmup_successes", 6) or 6))
        target_mastery = _clamp(success_rate * min(1.0, state.success_steps / warmup), 0.0, 1.0)
        update_rate = _clamp(spec.get("mastery", {}).get("update_rate", 0.30), 0.01, 1.0, default=0.30)
        state.mastery_estimate = _round4(state.mastery_estimate * (1.0 - update_rate) + target_mastery * update_rate)
        return {
            "schema_id": "skill_scaffold_result_observation/v2",
            "skill_id": skill_id,
            "phase_id": state.phase_id,
            "step_id": state.last_step_id,
            "external_teacher_not_ap_core": True,
            "selected_hint": selected_hint,
            "hinted_action_ids": sorted(hinted_ids),
            "selected_action_ids": sorted(selected_ids),
            "expected_evidence": sorted(expected),
            "evidence_hits": evidence_hits,
            "success_score": success_score,
            "succeeded": bool(succeeded),
            "success_steps": int(state.success_steps),
            "failure_steps": int(state.failure_steps),
            "mastery_estimate": _round4(state.mastery_estimate),
            "effective_strength_after": self.effective_strength(skill_id),
        }

    def phase_summary(self, skill_id: str, *, reason: str = "status") -> dict:
        spec = self._spec(skill_id)
        state = self._state(skill_id)
        phase = self._phase(skill_id, state.phase_id)
        return {
            "schema_id": "skill_scaffold_phase_summary/v2",
            "skill_id": skill_id,
            "title": spec.get("title", skill_id),
            "reason": reason,
            "enabled": bool(state.enabled),
            "phase_id": phase["phase_id"],
            "teacher_signal": dict(phase.get("teacher_signal", {}) or {}),
            "base_strength": _round4(state.base_strength),
            "phase_strength": _round4(phase.get("strength", 0.0)),
            "effective_strength": self.effective_strength(skill_id),
            "mastery_estimate": _round4(state.mastery_estimate),
            "success_steps": int(state.success_steps),
            "failure_steps": int(state.failure_steps),
            "guardrails": list(spec.get("guardrails", []) or []),
            "teacher_boundary": "external teacher, not AP core",
        }

    def teacher_off_boundary(self, skill_id: str) -> dict:
        state = self._state(skill_id)
        intervention = dict(state.last_intervention or {})
        return {
            "schema_id": "skill_scaffold_teacher_off_boundary/v2",
            "skill_id": skill_id,
            "phase_id": state.phase_id,
            "active": state.phase_id in {"teacher_off", "cold_retest"} or not state.enabled,
            "state_item_count": len(list(intervention.get("state_items", []) or [])),
            "action_bias_count": len(list(intervention.get("action_biases", []) or [])),
            "has_feedback": bool(intervention.get("feedback", {})),
            "passed": (
                (state.phase_id not in {"teacher_off", "cold_retest"} and state.enabled)
                or (
                    len(list(intervention.get("state_items", []) or [])) == 0
                    and len(list(intervention.get("action_biases", []) or [])) == 0
                    and not bool(intervention.get("feedback", {}))
                )
            ),
        }

    def effective_strength(self, skill_id: str) -> float:
        spec = self._spec(skill_id)
        state = self._state(skill_id)
        phase = self._phase(skill_id, state.phase_id)
        if not state.enabled:
            return 0.0
        phase_strength = _clamp(phase.get("strength", 0.0), 0.0, 1.0)
        decay = _clamp(spec.get("mastery", {}).get("decay_per_mastery", 0.72), 0.0, 0.95, default=0.72)
        return _round4(_clamp(state.base_strength, 0.0, 1.0) * phase_strength * (1.0 - decay * _clamp(state.mastery_estimate)))

    def _scaled_state_items(
        self,
        rows: list[dict],
        *,
        skill_id: str,
        step_id: str,
        phase_id: str,
        strength: float,
    ) -> list[dict]:
        scaled = []
        for index, row in enumerate(list(rows or [])):
            item = dict(row)
            energy = _clamp(item.get("real_energy", 0.18), 0.0, 1.0) * max(0.08, _clamp(strength))
            pressure = _clamp(item.get("cognitive_pressure", 0.06), 0.0, 1.0) * max(0.08, _clamp(strength))
            meta = dict(item.get("anchor_meta", {}) or {})
            meta.update(
                {
                    "skill_id": skill_id,
                    "step_id": step_id,
                    "phase_id": phase_id,
                    "source": "skill_scaffold_protocol_v2",
                    "meaning": "external teacher state material; first-class SA, not a hard gate",
                }
            )
            scaled.append(
                {
                    **item,
                    "sa_label": str(item.get("sa_label", "") or f"education_hint::{skill_id}::{step_id}::{index}"),
                    "family": str(item.get("family", "") or "education_intervention"),
                    "source_type": str(item.get("source_type", "") or "external_teacher"),
                    "real_energy": _round4(energy),
                    "cognitive_pressure": _round4(pressure),
                    "anchor_meta": meta,
                }
            )
        return scaled

    def _scaled_action_biases(
        self,
        rows: list[dict],
        *,
        skill_id: str,
        step_id: str,
        phase_id: str,
        strength: float,
    ) -> list[dict]:
        scaled = []
        for row in list(rows or []):
            bias = dict(row)
            action_id = str(bias.get("action_id", "") or "").strip()
            if not action_id:
                continue
            drive_delta = _round4(float(bias.get("drive_delta", bias.get("drive", 0.0)) or 0.0) * _clamp(strength))
            if abs(drive_delta) <= 0.00001:
                continue
            params = dict(bias.get("params", {}) or {})
            notes = _unique_notes(
                bias.get("notes", []),
                [
                    "Skill Scaffold Protocol v2",
                    "soft bias only",
                    "parameterized action hint" if params else "unparameterized action hint",
                    f"skill_id={skill_id}",
                    f"step_id={step_id}",
                    f"phase_id={phase_id}",
                ],
            )
            scaled.append(
                {
                    **bias,
                    "schema_id": "education_action_bias/v1",
                    "skill_id": skill_id,
                    "step_id": step_id,
                    "phase_id": phase_id,
                    "action_id": action_id,
                    "drive_delta": drive_delta,
                    "strength": _round4(strength),
                    "params": params,
                    "notes": notes,
                }
            )
        return scaled

    def _scaled_feedback(self, feedback: dict, *, phase_id: str, strength: float) -> dict:
        row = dict(feedback or {})
        if not row:
            return {}
        feedback_strength = max(0.12, _clamp(strength))
        return {
            "reward": _round4(_clamp(row.get("reward", 0.0)) * feedback_strength),
            "punishment": _round4(_clamp(row.get("punishment", 0.0)) * feedback_strength),
            "correctness": _round4(_clamp(row.get("correctness", 0.0)) * feedback_strength),
            "confidence": _round4(_clamp(row.get("confidence", 1.0), 0.0, 1.0, default=1.0)),
            "source": str(row.get("source", "") or "skill_scaffold_protocol_v2"),
            "notes": _unique_notes(row.get("notes", []), ["skill feedback", f"phase_id={phase_id}"]),
        }

    def _spec(self, skill_id: str) -> dict:
        key = str(skill_id or "").strip()
        if key not in self._specs:
            raise KeyError(f"Unknown skill spec: {skill_id}")
        return self._specs[key]

    def _state(self, skill_id: str) -> SkillProtocolStateV2:
        key = str(skill_id or "").strip()
        if key not in self._states:
            if key not in self._specs:
                raise KeyError(f"Unknown skill state: {skill_id}")
            first_phase = self._specs[key]["phases"][0]["phase_id"] if self._specs[key]["phases"] else "strong_scaffold"
            self._states[key] = SkillProtocolStateV2(skill_id=key, phase_id=first_phase)
        return self._states[key]

    def _phase(self, skill_id: str, phase_id: str) -> dict:
        spec = self._spec(skill_id)
        for row in list(spec.get("phases", []) or []):
            if row.get("phase_id") == phase_id:
                return dict(row)
        raise KeyError(f"Unknown phase {phase_id!r} for {skill_id}")

    def _step(self, skill_id: str, step_id: str) -> dict:
        spec = self._spec(skill_id)
        for row in list(spec.get("steps", []) or []):
            if row.get("step_id") == step_id:
                return dict(row)
        raise KeyError(f"Unknown step {step_id!r} for {skill_id}")
