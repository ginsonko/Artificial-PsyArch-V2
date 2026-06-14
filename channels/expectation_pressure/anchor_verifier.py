from __future__ import annotations

from dataclasses import dataclass, field


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass
class BAnchorState:
    anchor_id: str
    anchor_type: str
    source_memory_id: str
    source_memory_kind: str
    source_tick_index: int
    created_tick: int
    updated_tick: int
    source_branch: str
    expected_reward: float = 0.0
    expected_punishment: float = 0.0
    expected_correctness: float = 0.0
    expected_pressure: float = 0.0
    expected_virtual_energy: float = 0.0
    last_b_real_energy: float = 0.0
    last_b_virtual_energy: float = 0.0
    last_match_efficiency: float = 0.0
    level: float = 0.0
    age: int = 0
    support_count: int = 1
    verification_state: str = "active"
    evidence: dict = field(default_factory=dict)

    def as_trace(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "source_memory_id": self.source_memory_id,
            "source_memory_kind": self.source_memory_kind,
            "source_tick_index": int(self.source_tick_index),
            "created_tick": int(self.created_tick),
            "updated_tick": int(self.updated_tick),
            "source_branch": self.source_branch,
            "expected_reward": _round4(self.expected_reward),
            "expected_punishment": _round4(self.expected_punishment),
            "expected_correctness": _round4(self.expected_correctness),
            "expected_pressure": _round4(self.expected_pressure),
            "expected_virtual_energy": _round4(self.expected_virtual_energy),
            "last_b_real_energy": _round4(self.last_b_real_energy),
            "last_b_virtual_energy": _round4(self.last_b_virtual_energy),
            "last_match_efficiency": _round4(self.last_match_efficiency),
            "level": _round4(self.level),
            "age": int(self.age),
            "support_count": int(self.support_count),
            "verification_state": self.verification_state,
            "evidence": dict(self.evidence),
        }


class BAnchorExpectationVerifier:
    """
    Cross-tick verifier for expectation / pressure anchors.

    The anchor is a B object (a recalled memory), not a state-pool SA. Each tick
    compares active anchors against current Bn/Bn' rows, then emits small feeling
    SAs when the anchor is supported, missed, satisfied, or relieved.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_anchors: int = 32,
        decay: float = 0.88,
        min_anchor_level: float = 0.03,
        min_outcome_virtual: float = 0.045,
        validation_gain: float = 0.62,
        miss_gain: float = 0.34,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_anchors = max(1, int(max_anchors))
        self.decay = _clamp(decay, 0.0, 1.0)
        self.min_anchor_level = max(0.0, float(min_anchor_level))
        self.min_outcome_virtual = max(0.0, float(min_outcome_virtual))
        self.validation_gain = max(0.0, float(validation_gain))
        self.miss_gain = max(0.0, float(miss_gain))
        self._anchors: dict[str, BAnchorState] = {}
        self._last_tick = -1

    def update(
        self,
        *,
        tick_index: int,
        fast_bn: list[dict] | None = None,
        slow_bn: list[dict] | None = None,
        fast_cn: list[dict] | None = None,
        slow_cn: list[dict] | None = None,
        action_feedback_trace: dict | None = None,
        cognitive_feelings: dict | None = None,
    ) -> dict:
        tick = int(tick_index)
        self._advance_tick(tick)
        if not self.enabled:
            return {"schema_id": "expectation_pressure_b_anchor_verifier/v1", "enabled": False, "items": [], "anchors": []}

        current_b_rows = self._current_b_rows(fast_bn=fast_bn or [], slow_bn=slow_bn or [])
        created = self._create_or_refresh_anchors(
            tick_index=tick,
            fast_cn=fast_cn or [],
            slow_cn=slow_cn or [],
            current_b_rows=current_b_rows,
        )
        verified, missed, removed = self._verify_active_anchors(
            tick_index=tick,
            current_b_rows=current_b_rows,
            action_feedback_trace=action_feedback_trace or {},
            cognitive_feelings=cognitive_feelings or {},
        )
        items = self._build_items(tick_index=tick, verified=verified, missed=missed)
        active = sorted(
            [anchor.as_trace() for anchor in self._anchors.values()],
            key=lambda row: (-float(row.get("level", 0.0) or 0.0), str(row.get("anchor_id", ""))),
        )
        return {
            "schema_id": "expectation_pressure_b_anchor_verifier/v1",
            "enabled": True,
            "policy": {
                "anchor_semantics": "B_object_memory_id_not_state_pool_sa",
                "verification": "current_Bn_Bn_prime_real_energy_delta",
                "outcome_coupling": "reward_successor_creates_expectation;punishment_successor_creates_pressure",
                "bounded": True,
                "max_anchors": int(self.max_anchors),
            },
            "created": [anchor.as_trace() for anchor in created],
            "verified": [anchor.as_trace() for anchor in verified],
            "missed": [anchor.as_trace() for anchor in missed],
            "removed": removed,
            "items": items,
            "anchors": active[: self.max_anchors],
            "active_count": len(self._anchors),
        }

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(0, int(tick_index) - int(self._last_tick))
        if delta <= 0:
            return
        for anchor_id in list(self._anchors.keys()):
            anchor = self._anchors[anchor_id]
            anchor.level = _clamp(anchor.level * (self.decay**delta), 0.0, 1.0)
            anchor.age += delta
            if anchor.level < self.min_anchor_level:
                self._anchors.pop(anchor_id, None)
        self._last_tick = int(tick_index)

    def _current_b_rows(self, *, fast_bn: list[dict], slow_bn: list[dict]) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for branch, bn_rows in (("fast", fast_bn), ("slow", slow_bn)):
            for row in bn_rows or []:
                if not isinstance(row, dict):
                    continue
                memory_id = str(row.get("memory_id", "") or "")
                if not memory_id:
                    continue
                candidate = dict(row)
                candidate["branch"] = branch
                current_energy = self._b_real_energy(candidate)
                existing = rows.get(memory_id)
                if existing is None or current_energy > self._b_real_energy(existing):
                    rows[memory_id] = candidate
        return rows

    def _create_or_refresh_anchors(
        self,
        *,
        tick_index: int,
        fast_cn: list[dict],
        slow_cn: list[dict],
        current_b_rows: dict[str, dict],
    ) -> list[BAnchorState]:
        created_or_refreshed: list[BAnchorState] = []
        for branch, cn_rows in (("fast", fast_cn), ("slow", slow_cn)):
            for cn_row in cn_rows or []:
                if not isinstance(cn_row, dict):
                    continue
                source_memory_id = str(cn_row.get("source_memory_id", "") or "")
                if not source_memory_id:
                    continue
                outcome = self._successor_outcome(cn_row)
                if outcome["expected_reward"] < self.min_outcome_virtual and outcome["expected_punishment"] < self.min_outcome_virtual:
                    continue
                anchor_type = "pressure" if outcome["expected_punishment"] > outcome["expected_reward"] else "expectation"
                b_row = current_b_rows.get(source_memory_id, {})
                source_kind = str(b_row.get("memory_kind", "") or self._infer_memory_kind(cn_row))
                anchor_id = f"{anchor_type}:{source_kind}:{source_memory_id}"
                level = _clamp(outcome["expected_virtual_energy"] * (0.65 + self._b_match_efficiency(b_row) * 0.35), 0.0, 1.0)
                if level < self.min_anchor_level:
                    continue
                anchor = self._anchors.get(anchor_id)
                if anchor is None:
                    anchor = BAnchorState(
                        anchor_id=anchor_id,
                        anchor_type=anchor_type,
                        source_memory_id=source_memory_id,
                        source_memory_kind=source_kind,
                        source_tick_index=int(b_row.get("tick_index", -1) or -1),
                        created_tick=tick_index,
                        updated_tick=tick_index,
                        source_branch=branch,
                    )
                    self._anchors[anchor_id] = anchor
                else:
                    anchor.support_count += 1
                    anchor.updated_tick = tick_index
                    anchor.source_branch = branch
                anchor.expected_reward = max(anchor.expected_reward, outcome["expected_reward"])
                anchor.expected_punishment = max(anchor.expected_punishment, outcome["expected_punishment"])
                anchor.expected_correctness = max(anchor.expected_correctness, outcome["expected_correctness"])
                anchor.expected_pressure = max(anchor.expected_pressure, outcome["expected_pressure"])
                anchor.expected_virtual_energy = max(anchor.expected_virtual_energy, outcome["expected_virtual_energy"])
                anchor.last_b_real_energy = max(anchor.last_b_real_energy, self._b_real_energy(b_row))
                anchor.last_b_virtual_energy = max(anchor.last_b_virtual_energy, self._b_virtual_energy(b_row))
                anchor.last_match_efficiency = max(anchor.last_match_efficiency, self._b_match_efficiency(b_row))
                anchor.level = _clamp(max(anchor.level, level), 0.0, 1.0)
                anchor.verification_state = "active"
                anchor.evidence = {
                    "created_from": "successor_predicted_outcome",
                    "source_branch": branch,
                    "successor_memory_id": str(cn_row.get("successor_memory_id", "") or ""),
                    "expected_reward": _round4(outcome["expected_reward"]),
                    "expected_punishment": _round4(outcome["expected_punishment"]),
                    "expected_correctness": _round4(outcome["expected_correctness"]),
                    "expected_pressure": _round4(outcome["expected_pressure"]),
                }
                created_or_refreshed.append(anchor)
        self._trim_anchors()
        return created_or_refreshed

    def _verify_active_anchors(
        self,
        *,
        tick_index: int,
        current_b_rows: dict[str, dict],
        action_feedback_trace: dict,
        cognitive_feelings: dict,
    ) -> tuple[list[BAnchorState], list[BAnchorState], list[dict]]:
        verified: list[BAnchorState] = []
        missed: list[BAnchorState] = []
        removed: list[dict] = []
        feedback = dict((action_feedback_trace or {}).get("observed_feedback", {}) or {})
        reward = max(0.0, float(feedback.get("reward", 0.0) or 0.0))
        punishment = max(0.0, float(feedback.get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float(feedback.get("correctness", 0.0) or 0.0))
        channels = dict((cognitive_feelings or {}).get("channels", {}) or cognitive_feelings or {})
        correctness = max(correctness, max(0.0, float(channels.get("correctness", 0.0) or 0.0)) * 0.35)
        for anchor_id in list(self._anchors.keys()):
            anchor = self._anchors[anchor_id]
            if int(anchor.created_tick) == int(tick_index):
                continue
            current = current_b_rows.get(anchor.source_memory_id)
            previous_real = float(anchor.last_b_real_energy or 0.0)
            if current is None:
                miss = _clamp(anchor.level * self.miss_gain, 0.0, 1.0)
                anchor.level = _clamp(anchor.level - miss * 0.38, 0.0, 1.0)
                anchor.verification_state = "unverified_missing_b"
                anchor.updated_tick = tick_index
                anchor.evidence = {"reason": "B_not_recalled_this_tick", "miss_strength": _round4(miss)}
                missed.append(anchor)
            else:
                current_real = self._b_real_energy(current)
                current_virtual = self._b_virtual_energy(current)
                match = self._b_match_efficiency(current)
                delta_real = current_real - previous_real
                support = _clamp(max(0.0, delta_real) + current_real * 0.18 + match * 0.22, 0.0, 1.0)
                if anchor.anchor_type == "expectation":
                    outcome_support = _clamp(reward * 0.58 + correctness * 0.42, 0.0, 1.0)
                else:
                    outcome_support = _clamp(punishment * 0.62 + max(0.0, float(channels.get("pressure", 0.0) or 0.0)) * 0.26, 0.0, 1.0)
                if support > 0.035 or outcome_support > 0.035:
                    gain = _clamp((support + outcome_support) * self.validation_gain, 0.0, 1.0)
                    anchor.level = _clamp(anchor.level + gain * (1.0 - anchor.level * 0.35), 0.0, 1.0)
                    anchor.verification_state = "verified"
                    anchor.evidence = {
                        "reason": "B_recalled_and_supported",
                        "current_b_real_energy": _round4(current_real),
                        "delta_b_real_energy": _round4(delta_real),
                        "match_efficiency": _round4(match),
                        "outcome_support": _round4(outcome_support),
                    }
                    verified.append(anchor)
                elif delta_real < -0.035:
                    miss = _clamp(abs(delta_real) * self.miss_gain, 0.0, 1.0)
                    anchor.level = _clamp(anchor.level - miss * 0.34, 0.0, 1.0)
                    anchor.verification_state = "unverified_declining_b"
                    anchor.evidence = {
                        "reason": "B_real_energy_declined",
                        "current_b_real_energy": _round4(current_real),
                        "delta_b_real_energy": _round4(delta_real),
                        "miss_strength": _round4(miss),
                    }
                    missed.append(anchor)
                anchor.last_b_real_energy = current_real
                anchor.last_b_virtual_energy = current_virtual
                anchor.last_match_efficiency = match
                anchor.updated_tick = tick_index
            if anchor.level < self.min_anchor_level:
                removed.append({"anchor_id": anchor_id, "reason": "level_below_min", "level": _round4(anchor.level)})
                self._anchors.pop(anchor_id, None)
        return verified, missed, removed

    def _build_items(self, *, tick_index: int, verified: list[BAnchorState], missed: list[BAnchorState]) -> list[dict]:
        items: list[dict] = []
        for anchor in verified:
            energy = _clamp(anchor.level * 0.44 + anchor.expected_virtual_energy * 0.28, 0.0, 1.0)
            if energy < self.min_anchor_level:
                continue
            if anchor.anchor_type == "pressure":
                label = f"expectation_pressure::pressure_anchor_verified::{anchor.source_memory_id}"
                display = "压力锚验证"
            else:
                label = f"expectation_pressure::expectation_anchor_verified::{anchor.source_memory_id}"
                display = "期待锚验证"
            items.append(self._item(label, display, energy, anchor, tick_index, "verified"))
            if anchor.anchor_type == "expectation" and anchor.expected_reward > 0.0:
                items.append(self._item("feeling::satisfaction", "满足校验感", energy * 0.72, anchor, tick_index, "satisfaction"))
            if anchor.anchor_type == "pressure" and anchor.expected_punishment > 0.0:
                items.append(self._item("feeling::pressure_validation", "压力验证感", energy * 0.72, anchor, tick_index, "pressure_validation"))
        for anchor in missed:
            energy = _clamp(anchor.level * 0.36 + anchor.expected_virtual_energy * 0.18, 0.0, 1.0)
            if energy < self.min_anchor_level:
                continue
            label = f"expectation_pressure::anchor_gap::{anchor.source_memory_id}"
            items.append(self._item(label, "期待/压力锚落差", energy, anchor, tick_index, "gap"))
            if anchor.anchor_type == "expectation":
                items.append(self._item("feeling::expectation_gap", "期待落差", energy * 0.68, anchor, tick_index, "expectation_gap"))
        return items[:12]

    def _item(self, label: str, display: str, energy: float, anchor: BAnchorState, tick_index: int, event: str) -> dict:
        return {
            "sa_label": str(label),
            "display_text": str(display),
            "source_type": "expectation_pressure_anchor",
            "family": "expectation_pressure",
            "real_energy": _round4(max(0.0, float(energy))),
            "anchor_meta": {
                "schema_id": "expectation_pressure_b_anchor_item/v1",
                "tick_index": int(tick_index),
                "event": str(event),
                "anchor": anchor.as_trace(),
            },
        }

    def _successor_outcome(self, cn_row: dict) -> dict:
        reward = 0.0
        punishment = 0.0
        correctness = 0.0
        pressure = 0.0
        virtual_total = 0.0
        for item in list((cn_row or {}).get("predicted_items", []) or []):
            if not isinstance(item, dict):
                continue
            virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            if virtual <= 0.0:
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            observed = dict(anchor_meta.get("observed_feedback", {}) or {})
            feedback_semantics = dict(anchor_meta.get("feedback_energy_semantics", {}) or {})
            predicted_outcome = dict(anchor_meta.get("predicted_outcome", {}) or {})
            label_lower = label.lower()
            if label.startswith("action_feedback::") or family == "action_feedback" or source_type == "action_feedback":
                reward += virtual * max(0.0, float(observed.get("reward", 0.0) or 0.0) + float(observed.get("correctness", 0.0) or 0.0) * 0.35)
                punishment += virtual * max(0.0, float(observed.get("punishment", 0.0) or 0.0))
                correctness += virtual * max(0.0, float(observed.get("correctness", 0.0) or 0.0))
                pressure += virtual * max(0.0, float(feedback_semantics.get("punishment_pressure", 0.0) or 0.0))
            if label.startswith(("signal::reward", "rwd::", "reward::")):
                reward += virtual
            if "correctness" in label_lower or label == "feeling::correctness":
                correctness += virtual
            if label.startswith(("signal::punishment", "pun::", "punishment::")):
                punishment += virtual
            if label.startswith("expectation_pressure::pressure"):
                pressure += virtual * 0.72
            reward += virtual * max(0.0, float(predicted_outcome.get("reward", 0.0) or 0.0)) * 0.35
            punishment += virtual * max(0.0, float(predicted_outcome.get("punishment", 0.0) or 0.0)) * 0.35
            virtual_total += virtual
        return {
            "expected_reward": _clamp(reward, 0.0, 1.0),
            "expected_punishment": _clamp(punishment, 0.0, 1.0),
            "expected_correctness": _clamp(correctness, 0.0, 1.0),
            "expected_pressure": _clamp(pressure, 0.0, 1.0),
            "expected_virtual_energy": _clamp(max(reward + correctness * 0.35, punishment + pressure * 0.35, virtual_total * 0.08), 0.0, 1.0),
        }

    def _infer_memory_kind(self, cn_row: dict) -> str:
        source_id = str((cn_row or {}).get("source_memory_id", "") or "")
        if source_id.startswith("focus"):
            return "focus"
        return str((cn_row or {}).get("memory_kind", "") or "state")

    def _b_real_energy(self, row: dict | None) -> float:
        if not isinstance(row, dict):
            return 0.0
        return _clamp(
            max(
                float(row.get("b_real_energy", 0.0) or 0.0),
                float(row.get("b_effective_real_energy", 0.0) or 0.0),
                float(row.get("normalized_weight", 0.0) or 0.0) * float(row.get("match_efficiency", row.get("grasp_confidence", 0.0)) or 0.0),
            ),
            0.0,
            1.0,
        )

    def _b_virtual_energy(self, row: dict | None) -> float:
        if not isinstance(row, dict):
            return 0.0
        return _clamp(
            max(
                float(row.get("b_virtual_energy", 0.0) or 0.0),
                float(row.get("b_effective_virtual_energy", 0.0) or 0.0),
            ),
            0.0,
            1.0,
        )

    def _b_match_efficiency(self, row: dict | None) -> float:
        if not isinstance(row, dict):
            return 0.0
        return _clamp(float(row.get("match_efficiency", row.get("grasp_confidence", 0.0)) or 0.0), 0.0, 1.0)

    def _trim_anchors(self) -> None:
        if len(self._anchors) <= self.max_anchors:
            return
        ordered = sorted(self._anchors.values(), key=lambda anchor: (-float(anchor.level), -int(anchor.updated_tick), anchor.anchor_id))
        keep = {anchor.anchor_id for anchor in ordered[: self.max_anchors]}
        for anchor_id in list(self._anchors.keys()):
            if anchor_id not in keep:
                self._anchors.pop(anchor_id, None)
