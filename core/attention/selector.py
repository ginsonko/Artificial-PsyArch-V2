from __future__ import annotations

from heapq import nsmallest

"""
PHASE1_MINIMAL_UPGRADED:
Attention now includes a simple but explicit continuation bias so the slow
system does not depend only on the newest exogenous labels. It is still not the
final APV2.1 arbitration model, but it is stronger than the original one-shot
linear selector.
"""


class AttentionSelector:
    def __init__(
        self,
        *,
        focus_limit: int,
        pressure_gain: float,
        attention_gain_weight: float,
        fatigue_weight: float,
        continuation_bias: float = 0.35,
        real_energy_weight: float = 0.0,
        virtual_energy_weight: float = 0.25,
    ) -> None:
        self.focus_limit = max(1, int(focus_limit))
        self.pressure_gain = float(pressure_gain)
        self.attention_gain_weight = float(attention_gain_weight)
        self.fatigue_weight = float(fatigue_weight)
        self.continuation_bias = float(continuation_bias)
        self.real_energy_weight = max(0.0, float(real_energy_weight))
        self.virtual_energy_weight = max(0.0, float(virtual_energy_weight))

    def select(
        self,
        attention_rows: list[dict],
        *,
        previous_focus_labels: list[str] | None = None,
        emotion_modulation: dict | None = None,
        successor_bias: dict | None = None,
        innate_attention_biases: list[dict] | None = None,
        action_attention_controls: list[dict] | None = None,
    ) -> dict:
        previous_set = {str(label or "") for label in (previous_focus_labels or []) if str(label or "")}
        successor_bias_by_label = dict((successor_bias or {}).get("bias_by_label", {}) or {})
        innate_bias_by_label = self._build_innate_bias_by_label(innate_attention_biases or [])
        action_bias_by_label = self._build_action_bias_by_label(action_attention_controls or [])
        learned_band_by_label = self._build_learned_band_bias_by_label(action_attention_controls or [])
        # Extract emotion modulation from 8-channel NT system
        # emotion_modulation format: {"attention": {...}, "hdb": {...}, "action": {...}}
        attention_mod = (emotion_modulation or {}).get("attention", {})
        resource_multiplier = float(attention_mod.get("resource_multiplier", 1.0))
        threshold_adjustment = float(attention_mod.get("threshold_adjustment", 0.0))

        ranked = []
        for row in attention_rows:
            label = str(row.get("sa_label", "") or "")
            continuation_bonus = self.continuation_bias if label in previous_set else 0.0
            successor_bonus = max(0.0, float(successor_bias_by_label.get(label, 0.0) or 0.0))
            innate_bonus = float(innate_bias_by_label.get(label, 0.0) or 0.0)
            action_bias = dict(action_bias_by_label.get(label, {}) or {})
            action_boost = float(action_bias.get("boost", 0.0) or 0.0)
            action_suppression = float(action_bias.get("suppression", 0.0) or 0.0)
            learned_band = dict(learned_band_by_label.get(label, {}) or {})
            learned_band_boost = float(learned_band.get("boost", 0.0) or 0.0)
            learned_band_suppression = float(learned_band.get("suppression", 0.0) or 0.0)
            action_net_bias = action_boost - action_suppression
            learned_band_net_bias = learned_band_boost - learned_band_suppression
            base_score = (
                float(row.get("cognitive_pressure", 0.0) or 0.0) * self.pressure_gain
                + float(row.get("real_energy", 0.0) or 0.0) * self.real_energy_weight
                + float(row.get("attention_gain", 0.0) or 0.0) * self.attention_gain_weight
                + float(row.get("virtual_energy", 0.0) or 0.0) * self.virtual_energy_weight
                - float(row.get("fatigue", 0.0) or 0.0) * self.fatigue_weight
                + continuation_bonus
                + successor_bonus
                + innate_bonus
                + action_net_bias
                + learned_band_net_bias
            )
            # Apply emotion modulation to attention resource
            score = base_score * resource_multiplier
            enriched = dict(row)
            enriched["continuation_bonus"] = round(continuation_bonus, 4)
            enriched["successor_bias"] = round(successor_bonus, 4)
            enriched["innate_attention_bias"] = round(innate_bonus, 4)
            enriched["action_attention_boost"] = round(action_boost, 4)
            enriched["action_attention_suppression"] = round(action_suppression, 4)
            enriched["action_attention_net_bias"] = round(action_net_bias, 4)
            enriched["learned_band_boost"] = round(learned_band_boost, 4)
            enriched["learned_band_suppression"] = round(learned_band_suppression, 4)
            enriched["learned_band_net_bias"] = round(learned_band_net_bias, 4)
            enriched["learned_band_score"] = round(float(learned_band.get("score", 0.0) or 0.0), 4)
            enriched["learned_band_association_score"] = round(float(learned_band.get("association_score", 0.0) or 0.0), 4)
            enriched["learned_band_vector_score"] = round(float(learned_band.get("vector_score", 0.0) or 0.0), 4)
            enriched["real_energy_attention_weight"] = round(self.real_energy_weight, 4)
            enriched["virtual_energy_attention_weight"] = round(self.virtual_energy_weight, 4)
            if action_bias.get("sources"):
                enriched["action_attention_sources"] = list(action_bias.get("sources", []) or [])[:4]
            if learned_band.get("sources"):
                enriched["learned_band_sources"] = list(learned_band.get("sources", []) or [])[:4]
            if learned_band.get("anchor_tokens"):
                enriched["learned_band_anchor_tokens"] = list(learned_band.get("anchor_tokens", []) or [])[:8]
            enriched["base_focus_score"] = round(base_score, 4)
            enriched["emotion_multiplier"] = round(resource_multiplier, 4)
            enriched["focus_score"] = round(score, 4)
            ranked.append(enriched)
        rank_key = lambda item: (-float(item["focus_score"]), str(item["sa_label"]))
        ranked_limit = max(self.focus_limit * 8, self.focus_limit + 8)
        if len(ranked) > ranked_limit:
            ranked_items = nsmallest(ranked_limit, ranked, key=rank_key)
        else:
            ranked.sort(key=rank_key)
            ranked_items = ranked
        selected = ranked_items[: self.focus_limit]
        return {
            "selected_labels": [item["sa_label"] for item in selected],
            "selected_items": selected,
            "ranked_items": ranked_items,
            "innate_attention_biases": list(innate_attention_biases or [])[:8],
            "action_attention_controls": list(action_attention_controls or [])[:8],
        }

    def _build_innate_bias_by_label(self, biases: list[dict]) -> dict[str, float]:
        by_label: dict[str, float] = {}
        for bias in biases or []:
            if not isinstance(bias, dict):
                continue
            strength = max(0.0, float(bias.get("strength", 0.0) or 0.0))
            if strength <= 0.0:
                continue
            target_labels = [str(label or "") for label in list(bias.get("target_labels", []) or []) if str(label or "")]
            if not target_labels:
                anchor_key = str(bias.get("anchor_key", "") or "")
                if anchor_key and anchor_key != "global":
                    target_labels = [anchor_key]
            for label in target_labels:
                by_label[label] = by_label.get(label, 0.0) + min(0.42, strength * 0.35)
        return {label: round(min(0.6, value), 4) for label, value in by_label.items()}

    def _build_action_bias_by_label(self, controls: list[dict]) -> dict[str, dict]:
        by_label: dict[str, dict] = {}
        for control in controls or []:
            if not isinstance(control, dict):
                continue
            strength = max(0.0, float(control.get("strength", 0.0) or 0.0))
            if strength <= 0.0:
                continue
            source = str(control.get("source_action_id", "") or control.get("control_kind", "") or "action_control")
            control_kind = str(control.get("control_kind", "") or "")
            # Inspect/focus actions should be visible but bounded. They bias the
            # next readout of the cognitive field; they do not create truth.
            boost_gain = 0.55 if control_kind in {"focus_anchor", "inspect_residual"} else 0.38
            suppression_gain = 0.65 if control_kind == "release_focus" else 0.45
            for label in [str(item or "") for item in list(control.get("boost_labels", []) or []) if str(item or "")]:
                bucket = by_label.setdefault(label, {"boost": 0.0, "suppression": 0.0, "sources": []})
                bucket["boost"] = float(bucket.get("boost", 0.0) or 0.0) + min(0.55, strength * boost_gain)
                sources = list(bucket.get("sources", []) or [])
                if source not in sources:
                    sources.append(source)
                bucket["sources"] = sources
            for label in [str(item or "") for item in list(control.get("suppress_labels", []) or []) if str(item or "")]:
                bucket = by_label.setdefault(label, {"boost": 0.0, "suppression": 0.0, "sources": []})
                bucket["suppression"] = float(bucket.get("suppression", 0.0) or 0.0) + min(0.62, strength * suppression_gain)
                sources = list(bucket.get("sources", []) or [])
                if source not in sources:
                    sources.append(source)
                bucket["sources"] = sources
        return {
            label: {
                "boost": round(min(0.75, float(value.get("boost", 0.0) or 0.0)), 4),
                "suppression": round(min(0.75, float(value.get("suppression", 0.0) or 0.0)), 4),
                "sources": list(value.get("sources", []) or [])[:4],
            }
            for label, value in by_label.items()
        }

    def _build_learned_band_bias_by_label(self, controls: list[dict]) -> dict[str, dict]:
        by_label: dict[str, dict] = {}
        for control in controls or []:
            if not isinstance(control, dict):
                continue
            strength = max(0.0, float(control.get("strength", 0.0) or 0.0))
            if strength <= 0.0:
                continue
            control_kind = str(control.get("control_kind", "") or "")
            if control_kind == "diverge_attention":
                # Divergence means relaxing a learned band. It should not make
                # any learned neighborhood stronger in selector scoring.
                continue
            band_biases = list(control.get("learned_band_biases", []) or [])
            if not band_biases:
                continue
            source = str(control.get("source_action_id", "") or control_kind or "learned_band_control")
            band_mode = str(control.get("band_mode", "") or ("release" if control_kind == "release_focus" else "narrow"))
            band_gain = max(0.0, float(control.get("band_gain", 0.0) or 0.0))
            suppression_gain = max(0.0, float(control.get("band_suppression_gain", 0.0) or 0.0))
            if band_gain <= 0.0 and suppression_gain <= 0.0:
                band_gain = 0.42 if control_kind in {"focus_anchor", "inspect_residual"} else 0.30
                suppression_gain = 0.30 if band_mode == "narrow" else 0.0
            positive_scores = [
                max(0.0, float(row.get("score", 0.0) or 0.0))
                for row in band_biases
                if isinstance(row, dict)
            ]
            positive_scores = [score for score in positive_scores if score > 0.0]
            score_total = sum(positive_scores)
            top_band_score = max(positive_scores) if positive_scores else 0.0
            band_candidate_count = max(1, len(positive_scores))
            random_share = 1.0 / float(band_candidate_count)
            share_floor = random_share * (1.0 - random_share)
            relative_floor = random_share
            for row in band_biases:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("sa_label", "") or "")
                if not label:
                    continue
                score = max(0.0, float(row.get("score", 0.0) or 0.0))
                vector_score = float(row.get("vector_score", 0.0) or 0.0)
                association_score = float(row.get("association_score", 0.0) or 0.0)
                evidence_count = int(row.get("evidence_count", 0) or 0)
                feedback_utility = float(row.get("feedback_utility", 0.0) or 0.0)
                feedback_reward = max(0.0, float(row.get("feedback_reward", 0.0) or 0.0))
                feedback_correctness = max(0.0, float(row.get("feedback_correctness", 0.0) or 0.0))
                feedback_punishment = max(0.0, float(row.get("feedback_punishment", 0.0) or 0.0))
                score_share = score / score_total if score_total > 0.0 else 0.0
                relative_score = score / top_band_score if top_band_score > 0.0 else 0.0
                if score <= 0.0 and evidence_count <= 0:
                    continue
                bucket = by_label.setdefault(
                    label,
                    {
                        "boost": 0.0,
                        "suppression": 0.0,
                        "score": 0.0,
                        "vector_score": 0.0,
                        "association_score": 0.0,
                        "feedback_utility": 0.0,
                        "feedback_reward": 0.0,
                        "feedback_correctness": 0.0,
                        "feedback_punishment": 0.0,
                        "sources": [],
                        "anchor_tokens": [],
                    },
                )
                if band_mode == "release":
                    distribution_mass = max(score, score_total / float(band_candidate_count) if score_total > 0.0 else 0.0)
                    suppression = min(0.42, strength * max(0.08, suppression_gain or 0.22) * distribution_mass)
                    bucket["suppression"] = float(bucket.get("suppression", 0.0) or 0.0) + suppression
                else:
                    boost = min(0.52, strength * max(0.08, band_gain) * score)
                    bucket["boost"] = float(bucket.get("boost", 0.0) or 0.0) + boost
                    weak_by_distribution = bool(
                        band_mode == "narrow"
                        and evidence_count > 0
                        and score_total > 0.0
                        and score < top_band_score
                        and score_share < share_floor
                        and relative_score < relative_floor
                    )
                    if weak_by_distribution:
                        weakness = max(0.0, share_floor - score_share) + max(0.0, relative_floor - relative_score)
                        positive_feedback = max(0.0, feedback_utility, feedback_reward + feedback_correctness * 0.35 - feedback_punishment)
                        feedback_relief = positive_feedback / max(positive_feedback + weakness, 1e-6)
                        suppression = min(0.28, strength * max(0.04, suppression_gain) * weakness * (1.0 - feedback_relief))
                        bucket["suppression"] = float(bucket.get("suppression", 0.0) or 0.0) + suppression
                bucket["score"] = max(float(bucket.get("score", 0.0) or 0.0), score)
                bucket["vector_score"] = max(float(bucket.get("vector_score", 0.0) or 0.0), vector_score)
                bucket["association_score"] = max(float(bucket.get("association_score", 0.0) or 0.0), association_score)
                bucket["feedback_utility"] = max(float(bucket.get("feedback_utility", 0.0) or 0.0), feedback_utility)
                bucket["feedback_reward"] = max(float(bucket.get("feedback_reward", 0.0) or 0.0), feedback_reward)
                bucket["feedback_correctness"] = max(float(bucket.get("feedback_correctness", 0.0) or 0.0), feedback_correctness)
                bucket["feedback_punishment"] = max(float(bucket.get("feedback_punishment", 0.0) or 0.0), feedback_punishment)
                sources = list(bucket.get("sources", []) or [])
                if source not in sources:
                    sources.append(source)
                bucket["sources"] = sources
                anchors = list(bucket.get("anchor_tokens", []) or [])
                for token in list(row.get("anchor_tokens", []) or []):
                    clean = str(token or "")
                    if clean and clean not in anchors:
                        anchors.append(clean)
                bucket["anchor_tokens"] = anchors[:8]
        return {
            label: {
                "boost": round(min(0.7, float(value.get("boost", 0.0) or 0.0)), 4),
                "suppression": round(min(0.7, float(value.get("suppression", 0.0) or 0.0)), 4),
                "score": round(float(value.get("score", 0.0) or 0.0), 4),
                "vector_score": round(float(value.get("vector_score", 0.0) or 0.0), 4),
                "association_score": round(float(value.get("association_score", 0.0) or 0.0), 4),
                "feedback_utility": round(float(value.get("feedback_utility", 0.0) or 0.0), 4),
                "feedback_reward": round(float(value.get("feedback_reward", 0.0) or 0.0), 4),
                "feedback_correctness": round(float(value.get("feedback_correctness", 0.0) or 0.0), 4),
                "feedback_punishment": round(float(value.get("feedback_punishment", 0.0) or 0.0), 4),
                "sources": list(value.get("sources", []) or [])[:4],
                "anchor_tokens": list(value.get("anchor_tokens", []) or [])[:8],
            }
            for label, value in by_label.items()
        }
