from __future__ import annotations

import math
import random
from dataclasses import dataclass

from strict_core.common import round4
from strict_core.environments import CHOICE_ACTIONS, HOLD_ACTION


def numeric_vector(state_items: list[dict]) -> list[float]:
    for item in state_items or []:
        features = item.get("numeric_features", {})
        if isinstance(features, dict) and features:
            return [float(value[0] if isinstance(value, list) else value) for _, value in sorted(features.items())]
    return []


def positive_mass(values: list[float]) -> float:
    return sum(max(0.0, float(value)) for value in values)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    a = left[:size]
    b = right[:size]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


def euclidean_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dist = math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(size)))
    return 1.0 / (1.0 + dist)


def feature_overlap_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = 0.0
    denominator = 0.0
    for index in range(size):
        a = max(0.0, float(left[index]))
        b = max(0.0, float(right[index]))
        numerator += min(a, b)
        denominator += a
    if denominator <= 1e-9:
        return 0.0
    return numerator / denominator


@dataclass
class PolicyDecision:
    selected_action: str
    reason: str
    action_candidates: list[dict]
    grasp: float
    recalled_count: int


class ActionFeedbackMemoryPolicy:
    """Generic action policy driven by visible state and recalled feedback rows."""

    def __init__(
        self,
        *,
        choice_actions: tuple[str, ...] = CHOICE_ACTIONS,
        hold_action: str = HOLD_ACTION,
        explore_seed: int = 0,
        low_grasp_threshold: float = 0.38,
        hold_threshold: float = 0.42,
    ) -> None:
        self.choice_actions = tuple(choice_actions)
        self.hold_action = str(hold_action)
        self.rng = random.Random(int(explore_seed))
        self.low_grasp_threshold = float(low_grasp_threshold)
        self.hold_threshold = float(hold_threshold)
        self.step_index = 0

    def decide(self, *, state_items: list[dict], memory_rows: list[dict], learning_enabled: bool) -> PolicyDecision:
        self.step_index += 1
        current = numeric_vector(state_items)
        current_mass = positive_mass(current)
        estimates = {
            action: {"value": 0.0, "support": 0.0, "positive": 0.0, "negative": 0.0}
            for action in self.choice_actions
        }
        best_recall_similarity = 0.0
        used_rows = 0
        for row in memory_rows or []:
            snapshot = row.get("snapshot", {}) or {}
            items = list(snapshot.get("items", []) or [])
            previous = numeric_vector(items)
            previous_mass = positive_mass(previous)
            mass_ratio = min(current_mass, previous_mass) / max(1e-9, max(current_mass, previous_mass))
            if mass_ratio < 0.32:
                continue
            local_similarity = euclidean_similarity(current, previous)
            overlap_similarity = feature_overlap_similarity(current, previous)
            similarity = max(local_similarity, overlap_similarity * 0.92, cosine_similarity(current, previous) * 0.45) * mass_ratio
            if similarity <= 0.34:
                continue
            best_recall_similarity = max(best_recall_similarity, similarity)
            action_id = self._extract_action(items)
            feedback = self._extract_feedback(items, action_id=action_id)
            if action_id not in estimates or feedback is None:
                continue
            reward = float(feedback.get("reward", 0.0) or 0.0)
            punishment = float(feedback.get("punishment", 0.0) or 0.0)
            weight = (similarity ** 3) * max(0.1, min(3.0, float(row.get("score", 0.0) or 0.0)))
            estimates[action_id]["positive"] += weight * reward
            estimates[action_id]["negative"] += weight * punishment
            estimates[action_id]["support"] += weight
            used_rows += 1
        for action, data in estimates.items():
            data["value"] = (data["positive"] - data["negative"]) / max(1e-9, data["support"])
        best_action = max(self.choice_actions, key=lambda action: (estimates[action]["value"], estimates[action]["support"], action))
        sorted_values = sorted((estimates[action]["value"] for action in self.choice_actions), reverse=True)
        margin = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
        grasp = best_recall_similarity
        if used_rows == 0 or grasp < self.low_grasp_threshold:
            if learning_enabled:
                selected = self.choice_actions[self.step_index % len(self.choice_actions)]
                reason = "low_grasp_explore"
            else:
                selected = self.hold_action
                reason = "teacher_off_low_grasp_hold"
        elif learning_enabled and (self.step_index % len(self.choice_actions) == 0 or margin < 0.14):
            selected = self.choice_actions[self.step_index % len(self.choice_actions)]
            reason = "training_explore_to_collect_counterevidence"
        elif not learning_enabled and (estimates[best_action]["value"] <= 0.05 or margin < 0.012 or grasp < self.hold_threshold):
            selected = self.hold_action
            reason = "teacher_off_uncertain_hold"
        else:
            selected = best_action
            reason = "memory_supported_action"
        candidates = []
        for action in self.choice_actions:
            row = estimates[action]
            candidates.append(
                {
                    "action_id": action,
                    "value": round4(row["value"]),
                    "support": round4(row["support"]),
                    "positive": round4(row["positive"]),
                    "negative": round4(row["negative"]),
                }
            )
        candidates.append({"action_id": self.hold_action, "value": 0.0, "support": 0.0, "positive": 0.0, "negative": 0.0})
        candidates.sort(key=lambda item: (-float(item["value"]), -float(item["support"]), str(item["action_id"])))
        return PolicyDecision(
            selected_action=selected,
            reason=reason,
            action_candidates=candidates,
            grasp=round4(grasp),
            recalled_count=used_rows,
        )

    @staticmethod
    def _extract_action(items: list[dict]) -> str:
        for item in items:
            label = str(item.get("sa_label", "") or "")
            if label.startswith("action::"):
                return label
            meta = item.get("anchor_meta", {}) or {}
            action_id = str(meta.get("action_id", "") or "")
            if action_id.startswith("action::"):
                return action_id
        return ""

    @staticmethod
    def _extract_feedback(items: list[dict], *, action_id: str) -> dict | None:
        for item in items:
            if str(item.get("family", "") or "") != "action_feedback":
                continue
            meta = item.get("anchor_meta", {}) or {}
            if action_id and str(meta.get("action_id", "") or "") != action_id:
                continue
            return dict(meta)
        return None


class ExactPatternPolicy:
    """Control policy that can only reuse exact rounded whole-state patterns."""

    def __init__(self, *, choice_actions: tuple[str, ...] = CHOICE_ACTIONS, hold_action: str = HOLD_ACTION) -> None:
        self.choice_actions = tuple(choice_actions)
        self.hold_action = str(hold_action)
        self.table: dict[tuple[float, ...], dict[str, float]] = {}
        self.step_index = 0

    def decide(self, *, state_items: list[dict], memory_rows: list[dict], learning_enabled: bool) -> PolicyDecision:
        self.step_index += 1
        key = self._key(state_items)
        estimates = self.table.get(key, {})
        if not estimates:
            if learning_enabled:
                selected = self.choice_actions[self.step_index % len(self.choice_actions)]
                reason = "exact_pattern_explore"
            else:
                selected = self.hold_action
                reason = "exact_pattern_no_match_hold"
        else:
            selected = max(self.choice_actions, key=lambda action: (estimates.get(action, 0.0), action))
            reason = "exact_pattern_table_match"
        candidates = [
            {"action_id": action, "value": round4(estimates.get(action, 0.0)), "support": 1.0 if action in estimates else 0.0}
            for action in self.choice_actions
        ]
        candidates.append({"action_id": self.hold_action, "value": 0.0, "support": 0.0})
        return PolicyDecision(selected, reason, candidates, 1.0 if estimates else 0.0, len(estimates))

    def learn(self, *, state_items: list[dict], action_id: str, feedback: dict) -> None:
        key = self._key(state_items)
        table = self.table.setdefault(key, {})
        reward = float(feedback.get("reward", 0.0) or 0.0)
        punishment = float(feedback.get("punishment", 0.0) or 0.0)
        table[str(action_id)] = table.get(str(action_id), 0.0) + reward - punishment

    @staticmethod
    def _key(state_items: list[dict]) -> tuple[float, ...]:
        return tuple(round(value, 2) for value in numeric_vector(state_items))


class FeatureContributionPolicy:
    """Learns per-feature action contributions from action-feedback memory."""

    def __init__(
        self,
        *,
        choice_actions: tuple[str, ...] = CHOICE_ACTIONS,
        hold_action: str = HOLD_ACTION,
        explore_seed: int = 0,
    ) -> None:
        self.choice_actions = tuple(choice_actions)
        self.hold_action = str(hold_action)
        self.rng = random.Random(int(explore_seed))
        self.step_index = 0
        self.channel_scores: dict[int, dict[str, float]] = {}
        self.channel_support: dict[int, dict[str, float]] = {}

    def decide(self, *, state_items: list[dict], memory_rows: list[dict], learning_enabled: bool) -> PolicyDecision:
        self.step_index += 1
        channel_scores, channel_support = self._build_contributions(memory_rows)
        values = numeric_vector(state_items)
        estimates = {
            action: {"value": 0.0, "support": 0.0, "positive": 0.0, "negative": 0.0}
            for action in self.choice_actions
        }
        active_mass = sum(max(0.0, value) for value in values)
        for index, value in enumerate(values):
            strength = max(0.0, float(value))
            if strength <= 0.10:
                continue
            scores = channel_scores.get(index, {})
            supports = channel_support.get(index, {})
            for action in self.choice_actions:
                support = float(supports.get(action, 0.0) or 0.0)
                if support <= 0.0:
                    continue
                contribution = strength * float(scores.get(action, 0.0) or 0.0)
                estimates[action]["value"] += contribution
                estimates[action]["support"] += strength * support
                if contribution > 0:
                    estimates[action]["positive"] += contribution
                else:
                    estimates[action]["negative"] += abs(contribution)
        best_action = max(self.choice_actions, key=lambda action: (estimates[action]["value"], estimates[action]["support"], action))
        sorted_values = sorted([estimates[action]["value"] for action in self.choice_actions], reverse=True)
        margin = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else sorted_values[0]
        support_mass = max(estimates[action]["support"] for action in self.choice_actions)
        grasp = min(1.0, support_mass / max(0.1, active_mass))
        if support_mass <= 0.0:
            if learning_enabled:
                selected = self.choice_actions[self.step_index % len(self.choice_actions)]
                reason = "feature_contribution_explore"
            else:
                selected = self.hold_action
                reason = "teacher_off_no_feature_support_hold"
        elif learning_enabled and (self.step_index % len(self.choice_actions) == 0 or margin < 0.10):
            selected = self.choice_actions[self.step_index % len(self.choice_actions)]
            reason = "feature_training_explore_to_collect_counterevidence"
        elif not learning_enabled and (grasp < 0.35 or margin < 0.10 or estimates[best_action]["value"] <= 0.0):
            selected = self.hold_action
            reason = "teacher_off_low_feature_grasp_hold"
        else:
            selected = best_action
            reason = "feature_contribution_supported_action"
        candidates = []
        for action in self.choice_actions:
            row = estimates[action]
            candidates.append(
                {
                    "action_id": action,
                    "value": round4(row["value"]),
                    "support": round4(row["support"]),
                    "positive": round4(row["positive"]),
                    "negative": round4(row["negative"]),
                }
            )
        candidates.append({"action_id": self.hold_action, "value": 0.0, "support": 0.0, "positive": 0.0, "negative": 0.0})
        candidates.sort(key=lambda item: (-float(item["value"]), -float(item["support"]), str(item["action_id"])))
        return PolicyDecision(
            selected_action=selected,
            reason=reason,
            action_candidates=candidates,
            grasp=round4(grasp),
            recalled_count=sum(len(rows) for rows in self.channel_support.values()),
        )

    def _build_contributions(self, memory_rows: list[dict]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
        channel_scores: dict[int, dict[str, float]] = {}
        channel_support: dict[int, dict[str, float]] = {}
        for row in memory_rows or []:
            snapshot = row.get("snapshot", {}) or {}
            items = list(snapshot.get("items", []) or [])
            values = numeric_vector(items)
            action_id = ActionFeedbackMemoryPolicy._extract_action(items)
            feedback = ActionFeedbackMemoryPolicy._extract_feedback(items, action_id=action_id)
            if action_id not in self.choice_actions or feedback is None:
                continue
            reward = float(feedback.get("reward", 0.0) or 0.0)
            punishment = float(feedback.get("punishment", 0.0) or 0.0)
            signal = reward - punishment
            for index, value in enumerate(values):
                strength = max(0.0, float(value))
                if strength <= 0.10:
                    continue
                scores = channel_scores.setdefault(index, {})
                supports = channel_support.setdefault(index, {})
                scores[action_id] = scores.get(action_id, 0.0) + signal * strength
                supports[action_id] = supports.get(action_id, 0.0) + strength
        return channel_scores, channel_support
