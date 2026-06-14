from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _feature_value(features: dict, key: str) -> float:
    value = features.get(key, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp(float(value), 0.0, 1.0)
    if isinstance(value, str):
        return 1.0 if value.strip() else 0.0
    if isinstance(value, (list, tuple, set, dict)):
        return 1.0 if len(value) > 0 else 0.0
    return 0.0


def _weighted_sum(features: dict, weights: dict[str, float]) -> tuple[float, dict]:
    total = 0.0
    components = {}
    for key, weight in dict(weights or {}).items():
        value = _feature_value(features, key)
        contribution = value * float(weight)
        total += contribution
        components[key] = {
            "value": _round4(value),
            "weight": _round4(float(weight)),
            "contribution": _round4(contribution),
        }
    return total, components


@dataclass(frozen=True)
class CognitiveFeelingSpec:
    key: str
    display_text: str
    positive_features: dict[str, float] = field(default_factory=dict)
    negative_features: dict[str, float] = field(default_factory=dict)
    gain: float = 1.0
    min_activation: float = 0.12
    source_type: str = "cognitive_feeling_factory"
    policy: str = "non_answer_judge_soft_state_material"
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> "CognitiveFeelingSpec":
        row = dict(raw or {})
        key = str(row.get("key", "") or "").strip()
        if not key:
            raise ValueError("CognitiveFeelingSpec requires a key")
        return cls(
            key=key,
            display_text=str(row.get("display_text", "") or key),
            positive_features={str(k): float(v) for k, v in dict(row.get("positive_features", {}) or {}).items()},
            negative_features={str(k): float(v) for k, v in dict(row.get("negative_features", {}) or {}).items()},
            gain=max(0.0, float(row.get("gain", 1.0) or 1.0)),
            min_activation=max(0.0, float(row.get("min_activation", 0.12) or 0.12)),
            source_type=str(row.get("source_type", "") or "cognitive_feeling_factory"),
            policy=str(row.get("policy", "") or "non_answer_judge_soft_state_material"),
            notes=tuple(str(note) for note in list(row.get("notes", []) or []) if str(note or "").strip()),
        )


def default_feeling_factory_specs(*, min_activation: float = 0.12) -> list[CognitiveFeelingSpec]:
    return [
        CognitiveFeelingSpec(
            key="uncertainty",
            display_text="不确定感",
            min_activation=min_activation,
            positive_features={
                "mismatch_ratio": 0.32,
                "low_grasp": 0.28,
                "conflict_strength": 0.22,
                "ambiguity": 0.22,
                "residual_pressure": 0.18,
                "evidence_gap": 0.18,
            },
            negative_features={"grasp": 0.18, "step_expected_evidence_hit": 0.12, "sensory_clarity": 0.08},
            notes=("uncertainty raises evidence-seeking but does not decide answers",),
        ),
        CognitiveFeelingSpec(
            key="evidence_gap",
            display_text="证据缺口感",
            min_activation=min_activation,
            positive_features={
                "explicit_evidence_gap": 0.36,
                "missing_modality": 0.28,
                "conflict_strength": 0.22,
                "low_sensory_clarity": 0.22,
                "low_grasp": 0.18,
            },
            negative_features={"sensory_clarity": 0.16, "step_expected_evidence_hit": 0.12},
            notes=("evidence_gap is soft action material for resampling or recall",),
        ),
        CognitiveFeelingSpec(
            key="quantity_grasp",
            display_text="数量把握感",
            min_activation=min_activation,
            positive_features={
                "quantity_small_set_confidence": 0.44,
                "quantity_grounding": 0.32,
                "counting_progress": 0.22,
                "step_expected_evidence_hit": 0.12,
            },
            negative_features={"quantity_overflow": 0.32, "ambiguity": 0.14, "conflict_strength": 0.16},
            notes=("quantity_grasp is not a numeric answer",),
        ),
        CognitiveFeelingSpec(
            key="step_closure",
            display_text="步骤闭合感",
            min_activation=min_activation,
            positive_features={
                "step_expected_evidence_hit": 0.38,
                "action_feedback_success": 0.30,
                "grasp": 0.22,
                "low_mismatch": 0.18,
                "counting_progress": 0.10,
            },
            negative_features={"mismatch_ratio": 0.24, "residual_pressure": 0.16, "evidence_gap": 0.14},
            notes=("step_closure can be overturned by later dissonance or punishment",),
        ),
        CognitiveFeelingSpec(
            key="computation_pressure",
            display_text="计算压力",
            min_activation=min_activation,
            positive_features={
                "working_memory_load": 0.28,
                "operation_chain_length": 0.24,
                "quantity_overflow": 0.20,
                "residual_pressure": 0.18,
                "ambiguity": 0.16,
                "carry_borrow_pressure": 0.20,
            },
            negative_features={"step_closure_hint": 0.14, "grasp": 0.08},
            notes=("computation_pressure favors decomposition, reread, recompute, or wait",),
        ),
        CognitiveFeelingSpec(
            key="sensory_clarity",
            display_text="感官清晰感",
            min_activation=min_activation,
            positive_features={
                "visual_clarity": 0.28,
                "audio_clarity": 0.28,
                "focus_clarity": 0.22,
                "cross_modal_consistency": 0.20,
            },
            negative_features={"noise_level": 0.20, "missing_modality": 0.16, "conflict_strength": 0.12},
            notes=("sensory_clarity is confidence material, not a truth oracle",),
        ),
    ]


class CognitiveFeelingFactory:
    """Generic feature-to-feeling-SA factory.

    The factory is data-driven: domains add specs/features, not controller
    branches. It never chooses answers or actions. It only exposes bounded
    feeling SA so the normal AP state field can learn their consequences.
    """

    def __init__(self, specs: list[CognitiveFeelingSpec | dict] | None = None, *, min_activation: float = 0.12) -> None:
        self.specs = [
            spec if isinstance(spec, CognitiveFeelingSpec) else CognitiveFeelingSpec.from_dict(dict(spec))
            for spec in (specs if specs is not None else default_feeling_factory_specs(min_activation=min_activation))
        ]

    def derive(
        self,
        *,
        features: dict,
        source: str = "cognitive_feeling_factory",
        tick_index: int | None = None,
    ) -> dict:
        features = dict(features or {})
        channels: dict[str, float] = {}
        items: list[dict] = []
        traces: dict[str, dict] = {}
        for spec in self.specs:
            positive, positive_components = _weighted_sum(features, spec.positive_features)
            negative, negative_components = _weighted_sum(features, spec.negative_features)
            value = _clamp((positive - negative) * spec.gain, 0.0, 1.0)
            value = _round4(value)
            channels[spec.key] = value
            traces[spec.key] = {
                "positive": _round4(positive),
                "negative": _round4(negative),
                "gain": _round4(spec.gain),
                "value": value,
                "positive_components": positive_components,
                "negative_components": negative_components,
                "policy": spec.policy,
                "notes": list(spec.notes),
            }
            if value >= spec.min_activation:
                items.append(
                    {
                        "sa_label": f"feeling::{spec.key}",
                        "display_text": spec.display_text,
                        "source_type": spec.source_type,
                        "family": "cognitive_feeling",
                        "real_energy": value,
                        "cognitive_pressure": value,
                        "anchor_meta": {
                            "schema_id": "cognitive_feeling_factory_item/v1",
                            "feeling_key": spec.key,
                            "feeling_value": value,
                            "source": source,
                            "tick_index": tick_index,
                            "policy": spec.policy,
                            "meaning": "soft_state_material_not_answer_or_action_judge",
                            "non_answer_judge": True,
                            "soft_state_material": True,
                            "components": traces[spec.key],
                        },
                    }
                )
        return {
            "schema_id": "cognitive_feeling_factory_trace/v1",
            "source": source,
            "tick_index": tick_index,
            "policy": "generic_feature_to_feeling_sa_soft_bias_not_answer_judge",
            "channels": channels,
            "items": items,
            "features": {str(key): _feature_value(features, str(key)) for key in sorted(features)},
            "spec_traces": traces,
        }


def _weighted_grasp(rows: list[dict]) -> float:
    total = 0.0
    weighted = 0.0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        confidence = float(row.get("grasp_confidence", row.get("match_efficiency", 0.0)) or 0.0)
        if confidence <= 0.0:
            continue
        weight = float(row.get("normalized_weight", 0.0) or 0.0)
        if weight <= 0.0:
            weight = min(1.0, max(0.0, float(row.get("score", 0.0) or 0.0) / 3.0))
        weighted += confidence * weight
        total += weight
    if total <= 1e-9:
        return 0.0
    return _clamp(weighted / total, 0.0, 1.0)


class CognitiveFeelingChannel:
    def __init__(
        self,
        *,
        min_activation: float,
        surprise_gain: float,
        coherence_gain: float,
        dissonance_gain: float,
        correctness_gain: float,
        grasp_gain: float,
        expectation_gain: float,
        pressure_gain: float,
    ) -> None:
        self.min_activation = max(0.0, float(min_activation))
        self.surprise_gain = float(surprise_gain)
        self.coherence_gain = float(coherence_gain)
        self.dissonance_gain = float(dissonance_gain)
        self.correctness_gain = float(correctness_gain)
        self.grasp_gain = float(grasp_gain)
        self.expectation_gain = float(expectation_gain)
        self.pressure_gain = float(pressure_gain)
        self.factory = CognitiveFeelingFactory(min_activation=self.min_activation)

    def derive(
        self,
        *,
        state_snapshot_items: list[dict],
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        prediction_trace: dict | None = None,
        residual_summary: dict | None = None,
    ) -> dict:
        total_real = sum(float(item.get("real_energy", 0.0) or 0.0) for item in state_snapshot_items)
        total_virtual = sum(float(item.get("virtual_energy", 0.0) or 0.0) for item in state_snapshot_items)
        predicted_item_count = sum(len(branch.get("predicted_items", []) or []) for branch in fast_cn) + sum(
            len(branch.get("predicted_items", []) or []) for branch in slow_cn
        )
        fast_top = float(fast_bn[0]["score"]) if fast_bn else 0.0
        slow_top = float(slow_bn[0]["score"]) if slow_bn else 0.0
        fast_grasp = _weighted_grasp(fast_bn)
        slow_grasp = _weighted_grasp(slow_bn)
        fast_mass = max(0.0, total_real + total_virtual)
        pressure_ratio = total_virtual / max(1.0, fast_mass)
        overprediction = max(0.0, total_virtual - total_real)
        coherence = _clamp((fast_top * 0.55 + slow_top * 0.45) / 3.0, 0.0, 1.0)
        trace = dict(prediction_trace or {})
        residual = dict(residual_summary or {})
        mismatch_ratio = float(trace.get("mismatch_ratio", 0.0) or 0.0)
        alignment_score = float(trace.get("alignment_score", 0.0) or 0.0)
        unexpected_count = int(trace.get("unexpected_count", len(trace.get("unexpected_labels", []) or [])) or 0)
        missed_count = int(trace.get("missed_count", len(trace.get("missed_predicted_labels", []) or [])) or 0)
        residual_mass = float(residual.get("total_unresolved_mass", 0.0) or 0.0)
        residual_pressure = _clamp(residual_mass / max(1.0, total_real + total_virtual + residual_mass), 0.0, 1.0)
        evidence_gap_strength = 0.0
        conflict_strength = 0.0
        missing_modality = 0.0
        visual_clarity_values = []
        audio_clarity_values = []
        focus_clarity_values = []
        noise_values = []
        quantity_confidence_values = []
        quantity_grounding_values = []
        quantity_overflow = 0.0
        counting_progress = 0.0
        working_memory_load = 0.0
        operation_chain_length = 0.0
        carry_borrow_pressure = 0.0
        step_expected_evidence_hit = 0.0
        action_feedback_success = 0.0
        cross_modal_consistency = 0.0
        for item in state_snapshot_items or []:
            if not isinstance(item, dict):
                continue
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            label = str(item.get("sa_label", "") or "")
            meta = dict(item.get("anchor_meta", {}) or {})
            schema_id = str(meta.get("schema_id", "") or "")
            item_strength = _clamp(
                max(
                    float(item.get("real_energy", 0.0) or 0.0),
                    float(item.get("virtual_energy", 0.0) or 0.0),
                    abs(float(item.get("cognitive_pressure", 0.0) or 0.0)),
                    float(meta.get("strength", 0.0) or 0.0),
                ),
                0.0,
                1.0,
            )
            if family in {"evidence_gap", "uncertainty_evidence_gap"} or label.startswith("evidence_gap::") or schema_id in {"evidence_gap/v1", "uncertainty_evidence_gap/v1"}:
                evidence_gap_strength = max(evidence_gap_strength, item_strength)
                if meta.get("missing_modalities"):
                    missing_modality = 1.0
            if family in {"evidence_conflict", "modality_conflict"} or label.startswith("evidence_conflict::") or schema_id in {"evidence_conflict/v1", "modality_conflict/v1"}:
                conflict_strength = max(conflict_strength, item_strength)
            if family in {"vision", "vision_scene", "vision_object"} or source_type.startswith("vision"):
                if "clarity" in meta:
                    visual_clarity_values.append(_clamp(float(meta.get("clarity", 0.0) or 0.0), 0.0, 1.0))
                if "noise" in meta:
                    noise_values.append(_clamp(float(meta.get("noise", 0.0) or 0.0), 0.0, 1.0))
            if family in {"audio", "audio_event", "audio_semantic"} or source_type.startswith("audio"):
                if "clarity" in meta:
                    audio_clarity_values.append(_clamp(float(meta.get("clarity", 0.0) or 0.0), 0.0, 1.0))
                if "noise" in meta:
                    noise_values.append(_clamp(float(meta.get("noise", 0.0) or 0.0), 0.0, 1.0))
            if "focus_clarity" in meta:
                focus_clarity_values.append(_clamp(float(meta.get("focus_clarity", 0.0) or 0.0), 0.0, 1.0))
            if "quantity_small_set_confidence" in meta:
                quantity_confidence_values.append(_clamp(float(meta.get("quantity_small_set_confidence", 0.0) or 0.0), 0.0, 1.0))
            if "quantity_grounding" in meta:
                quantity_grounding_values.append(_clamp(float(meta.get("quantity_grounding", 0.0) or 0.0), 0.0, 1.0))
            if "quantity_overflow" in meta:
                quantity_overflow = max(quantity_overflow, _clamp(float(meta.get("quantity_overflow", 0.0) or 0.0), 0.0, 1.0))
            if "counting_progress" in meta:
                counting_progress = max(counting_progress, _clamp(float(meta.get("counting_progress", 0.0) or 0.0), 0.0, 1.0))
            if "working_memory_load" in meta:
                working_memory_load = max(working_memory_load, _clamp(float(meta.get("working_memory_load", 0.0) or 0.0), 0.0, 1.0))
            if "operation_chain_length" in meta:
                operation_chain_length = max(operation_chain_length, _clamp(float(meta.get("operation_chain_length", 0.0) or 0.0), 0.0, 1.0))
            if "carry_borrow_pressure" in meta:
                carry_borrow_pressure = max(carry_borrow_pressure, _clamp(float(meta.get("carry_borrow_pressure", 0.0) or 0.0), 0.0, 1.0))
            if "step_expected_evidence_hit" in meta:
                step_expected_evidence_hit = max(step_expected_evidence_hit, _clamp(float(meta.get("step_expected_evidence_hit", 0.0) or 0.0), 0.0, 1.0))
            if "action_feedback_success" in meta:
                action_feedback_success = max(action_feedback_success, _clamp(float(meta.get("action_feedback_success", 0.0) or 0.0), 0.0, 1.0))
            if "cross_modal_consistency" in meta:
                cross_modal_consistency = max(cross_modal_consistency, _clamp(float(meta.get("cross_modal_consistency", 0.0) or 0.0), 0.0, 1.0))

        surprise = _clamp(
            (predicted_item_count / max(1, len(state_snapshot_items) or 1)) * 0.35
            + overprediction * 0.08
            + mismatch_ratio * 0.42
            + min(1.0, unexpected_count / 4.0) * 0.18,
            0.0,
            1.0,
        )
        dissonance = _clamp(
            overprediction * 0.1
            + max(0.0, 0.55 - coherence)
            + mismatch_ratio * 0.52
            + residual_pressure * 0.38,
            0.0,
            1.0,
        )
        correctness = _clamp(coherence * 0.92 + alignment_score * 0.28 - dissonance * 0.3 - mismatch_ratio * 0.18, 0.0, 1.0)
        if fast_grasp > 0.0 or slow_grasp > 0.0:
            grasp = _clamp(slow_grasp * 0.65 + fast_grasp * 0.35, 0.0, 1.0)
        else:
            grasp = _clamp((slow_top * 0.65 + fast_top * 0.35) / 3.0, 0.0, 1.0)
        expectation = _clamp(
            (predicted_item_count / max(1, len(state_snapshot_items) or 1)) * 0.22
            + correctness * 0.35
            + alignment_score * 0.16
            - mismatch_ratio * 0.1,
            0.0,
            1.0,
        )
        pressure = _clamp(pressure_ratio * 0.9 + dissonance * 0.45 + residual_pressure * 0.35 + min(1.0, missed_count / 4.0) * 0.12, 0.0, 1.0)
        feelings = {
            "surprise": _round4(_clamp(surprise * self.surprise_gain, 0.0, 1.0)),
            "coherence": _round4(_clamp(coherence * self.coherence_gain, 0.0, 1.0)),
            "dissonance": _round4(_clamp(dissonance * self.dissonance_gain, 0.0, 1.0)),
            "correctness": _round4(_clamp(correctness * self.correctness_gain, 0.0, 1.0)),
            "grasp": _round4(_clamp(grasp * self.grasp_gain, 0.0, 1.0)),
            "expectation": _round4(_clamp(expectation * self.expectation_gain, 0.0, 1.0)),
            "pressure": _round4(_clamp(pressure * self.pressure_gain, 0.0, 1.0)),
        }
        factory_features = {
            "mismatch_ratio": mismatch_ratio,
            "low_mismatch": max(0.0, 1.0 - mismatch_ratio),
            "residual_pressure": residual_pressure,
            "ambiguity": _clamp(mismatch_ratio * 0.58 + min(1.0, missed_count / 4.0) * 0.20 + conflict_strength * 0.20, 0.0, 1.0),
            "grasp": feelings["grasp"],
            "low_grasp": max(0.0, 1.0 - feelings["grasp"]),
            "conflict_strength": conflict_strength,
            "explicit_evidence_gap": evidence_gap_strength,
            "evidence_gap": evidence_gap_strength,
            "missing_modality": missing_modality,
            "quantity_small_set_confidence": max(quantity_confidence_values or [0.0]),
            "quantity_grounding": max(quantity_grounding_values or [0.0]),
            "quantity_overflow": quantity_overflow,
            "counting_progress": counting_progress,
            "step_expected_evidence_hit": max(step_expected_evidence_hit, alignment_score),
            "action_feedback_success": action_feedback_success,
            "working_memory_load": working_memory_load,
            "operation_chain_length": operation_chain_length,
            "carry_borrow_pressure": carry_borrow_pressure,
            "visual_clarity": max(visual_clarity_values or [0.0]),
            "audio_clarity": max(audio_clarity_values or [0.0]),
            "focus_clarity": max(focus_clarity_values or [0.0]),
            "cross_modal_consistency": cross_modal_consistency,
            "noise_level": max(noise_values or [0.0]),
        }
        factory_features["low_sensory_clarity"] = max(
            0.0,
            1.0
            - max(
                factory_features["visual_clarity"],
                factory_features["audio_clarity"],
                factory_features["focus_clarity"],
                factory_features["cross_modal_consistency"],
            ),
        )
        factory_features["step_closure_hint"] = max(factory_features["step_expected_evidence_hit"], action_feedback_success)
        factory_trace = self.factory.derive(features=factory_features, source="cognitive_feeling_channel")
        for key, value in dict(factory_trace.get("channels", {}) or {}).items():
            feelings[key] = _round4(max(float(feelings.get(key, 0.0) or 0.0), float(value or 0.0)))
        items = []
        label_map = {
            "surprise": ("feeling::surprise", "惊"),
            "coherence": ("feeling::coherence", "合理感"),
            "dissonance": ("feeling::dissonance", "违和感"),
            "correctness": ("feeling::correctness", "正确感"),
            "grasp": ("feeling::grasp", "把握感"),
            "expectation": ("feeling::expectation", "期待"),
            "pressure": ("feeling::pressure", "压力"),
        }
        for key, value in feelings.items():
            if value < self.min_activation:
                continue
            if key not in label_map:
                continue
            label, display = label_map[key]
            items.append(
                {
                    "sa_label": label,
                    "display_text": display,
                    "source_type": "cognitive_feeling",
                    "family": "cognitive_feeling",
                    "real_energy": value,
                    "anchor_meta": {
                        "feeling_key": key,
                        "feeling_value": value,
                        "derived_from": {
                            "fast_top": _round4(fast_top),
                            "slow_top": _round4(slow_top),
                            "fast_grasp": _round4(fast_grasp),
                            "slow_grasp": _round4(slow_grasp),
                            "predicted_item_count": int(predicted_item_count),
                            "total_real": _round4(total_real),
                            "total_virtual": _round4(total_virtual),
                            "prediction_alignment_score": _round4(alignment_score),
                            "prediction_mismatch_ratio": _round4(mismatch_ratio),
                            "prediction_unexpected_count": int(unexpected_count),
                            "prediction_missed_count": int(missed_count),
                            "residual_pressure": _round4(residual_pressure),
                        },
                    },
                }
            )
        factory_item_labels = {str(item.get("sa_label", "") or "") for item in items}
        for item in list(factory_trace.get("items", []) or []):
            label = str(item.get("sa_label", "") or "")
            if label and label not in factory_item_labels:
                items.append(dict(item))
                factory_item_labels.add(label)
        return {
            "channels": feelings,
            "items": items,
            "factory": factory_trace,
            "prediction_coupling": {
                "alignment_score": _round4(alignment_score),
                "mismatch_ratio": _round4(mismatch_ratio),
                "unexpected_count": int(unexpected_count),
                "missed_count": int(missed_count),
                "residual_unresolved_mass": _round4(residual_mass),
                "residual_pressure": _round4(residual_pressure),
            },
        }
