from __future__ import annotations

import random
from dataclasses import dataclass

from strict_core.common import SECRET_RULE_TAG, SECRET_TRUTH_TAG, round4, stable_seed


CHOICE_ACTIONS = ("action::press_left", "action::press_right", "action::press_up")
HOLD_ACTION = "action::hold"


@dataclass(frozen=True)
class StrictCase:
    case_id: str
    split: str
    visible_state_items: list[dict]
    private_group: int | None = None
    private_values: list[float] | None = None
    judge_unknown: bool = False
    truth_taint: str = SECRET_TRUTH_TAG


class BlindButtonWorld:
    """Environment-only numeric pattern task.

    The world owns a private group->action mapping.  It exposes only numeric
    features and scores actions after an actuator event has been committed.
    """

    def __init__(self, *, seed: int, dimension: int = 8, group_count: int = 3) -> None:
        self.seed = int(seed)
        self.dimension = int(dimension)
        self.group_count = int(group_count)
        rng = random.Random(self.seed)
        self._centers = self._make_centers(dimension=self.dimension, group_count=self.group_count)
        actions = list(CHOICE_ACTIONS)
        rng.shuffle(actions)
        self._private_action_by_group = {index: actions[index] for index in range(self.group_count)}
        self._private_taint = SECRET_RULE_TAG

    def case_stream(self, *, split: str, count: int, start_variant: int = 0) -> list[StrictCase]:
        rows: list[StrictCase] = []
        for index in range(int(count)):
            group = index % self.group_count
            rows.append(self.make_case(split=split, index=index, group=group, variant=start_variant + index))
        rng = random.Random(stable_seed(self.seed, split, count, start_variant))
        rng.shuffle(rows)
        return rows

    def ood_stream(self, *, split: str, count: int, start_variant: int = 0) -> list[StrictCase]:
        rows: list[StrictCase] = []
        for index in range(int(count)):
            rng = random.Random(stable_seed(self.seed, split, "ood", start_variant, index))
            values = [round4(rng.uniform(0.00, 0.08)) for _ in range(self.dimension)]
            rows.append(
                StrictCase(
                    case_id=f"{split}_{index:03d}",
                    split=split,
                    visible_state_items=[self._state_item(values)],
                    private_group=None,
                    private_values=values,
                    judge_unknown=True,
                )
            )
        return rows

    def make_case(self, *, split: str, index: int, group: int, variant: int) -> StrictCase:
        rng = random.Random(stable_seed(self.seed, split, group, variant))
        center = self._centers[int(group)]
        values = [round4(value + rng.uniform(-0.052, 0.052)) for value in center]
        return StrictCase(
            case_id=f"{split}_{index:03d}",
            split=split,
            visible_state_items=[self._state_item(values)],
            private_group=int(group),
            private_values=values,
        )

    def judge(self, case: StrictCase, action_id: str) -> dict:
        if case.judge_unknown:
            rewarded = str(action_id) == HOLD_ACTION
            verdict = "held_unknown" if rewarded else "acted_on_unknown"
        else:
            rewarded = str(action_id) == self._private_action_by_group[int(case.private_group or 0)]
            verdict = "rewarded" if rewarded else "punished"
        return {
            "verdict": verdict,
            "reward": 1.0 if rewarded else 0.0,
            "punishment": 0.0 if rewarded else 1.0,
            "quality_tags": ["external_score_after_commit"],
            "answer_payload": None,
        }

    def perform_action(self, *, case: StrictCase, action_id: str) -> dict:
        return {
            "event_type": "external_actuator_event",
            "actuator_schema": "strict_blind_button_world/v1",
            "case_id": case.case_id,
            "executed_action_id": str(action_id),
            "execution_result": "committed",
            "environment_changed": True,
        }

    @staticmethod
    def inverted_feedback(feedback: dict) -> dict:
        rewarded = not (float(feedback.get("reward", 0.0) or 0.0) > 0.0)
        return {
            "verdict": "rewarded" if rewarded else "punished",
            "reward": 1.0 if rewarded else 0.0,
            "punishment": 0.0 if rewarded else 1.0,
            "quality_tags": ["control_feedback_counterfactual_to_current_commit"],
            "answer_payload": None,
        }

    @staticmethod
    def random_feedback(*, rng: random.Random) -> dict:
        rewarded = rng.random() < (1.0 / 3.0)
        return {
            "verdict": "rewarded" if rewarded else "punished",
            "reward": 1.0 if rewarded else 0.0,
            "punishment": 0.0 if rewarded else 1.0,
            "quality_tags": ["control_feedback_not_linked_to_current_commit"],
            "answer_payload": None,
        }

    @staticmethod
    def _state_item(values: list[float]) -> dict:
        return {
            "sa_label": "sensor::strict_blind_numeric_pattern",
            "display_text": "strict numeric pattern",
            "source_type": "strict_numeric_sensor",
            "family": "numeric_sensor",
            "real_energy": 1.0,
            "numeric_features": {f"blind.v{i}": round4(value) for i, value in enumerate(values)},
            "anchor_meta": {
                "schema_id": "strict_blind_numeric_sensor/v1",
                "numeric_only": True,
            },
        }

    @staticmethod
    def _make_centers(*, dimension: int, group_count: int) -> list[list[float]]:
        basis = [
            [0.86, 0.12, 0.18, 0.77, 0.20, 0.71, 0.31, 0.88],
            [0.18, 0.84, 0.72, 0.16, 0.88, 0.22, 0.78, 0.28],
            [0.69, 0.66, 0.24, 0.31, 0.37, 0.84, 0.82, 0.13],
        ]
        rows = [row[:dimension] for row in basis[:group_count]]
        rng = random.Random(991)
        while len(rows) < group_count:
            rows.append([rng.random() for _ in range(dimension)])
        return rows


class CompositionalFeatureWorld:
    """Environment-only feature contribution task.

    Training exposes single active channels.  Holdout combines two weak useful
    channels with one stronger distractor, so exact whole-pattern binding should
    fail while contribution learning can generalize.
    """

    def __init__(self, *, seed: int, dimension: int = 8) -> None:
        self.seed = int(seed)
        self.dimension = int(dimension)
        rng = random.Random(self.seed)
        channels = list(range(6))
        rng.shuffle(channels)
        actions = list(CHOICE_ACTIONS)
        rng.shuffle(actions)
        self._private_channels_by_action = {
            actions[0]: channels[0:2],
            actions[1]: channels[2:4],
            actions[2]: channels[4:6],
        }
        self._private_taint = SECRET_RULE_TAG

    def training_stream(self, *, split: str, repeat: int, start_variant: int = 0) -> list[StrictCase]:
        rows: list[StrictCase] = []
        index = 0
        for cycle in range(int(repeat)):
            for channel in range(6):
                rows.append(self.make_case(split=split, index=index, active_channels={channel: 0.88}, variant=start_variant + cycle * 100 + channel))
                index += 1
        rng = random.Random(stable_seed(self.seed, split, repeat, start_variant))
        rng.shuffle(rows)
        return rows

    def compositional_holdout_stream(self, *, split: str, repeat: int, start_variant: int = 0) -> list[StrictCase]:
        rows: list[StrictCase] = []
        index = 0
        for cycle in range(int(repeat)):
            for action in CHOICE_ACTIONS:
                useful = list(self._private_channels_by_action[action])
                distractor_actions = [candidate for candidate in CHOICE_ACTIONS if candidate != action]
                distractor_action = distractor_actions[(cycle + CHOICE_ACTIONS.index(action)) % len(distractor_actions)]
                distractor_channel = self._private_channels_by_action[distractor_action][cycle % 2]
                active = {useful[0]: 0.74, useful[1]: 0.74, distractor_channel: 0.84}
                rows.append(self.make_case(split=split, index=index, active_channels=active, variant=start_variant + cycle * 100 + index))
                index += 1
        rng = random.Random(stable_seed(self.seed, split, repeat, start_variant))
        rng.shuffle(rows)
        return rows

    def ood_stream(self, *, split: str, count: int, start_variant: int = 0) -> list[StrictCase]:
        rows: list[StrictCase] = []
        for index in range(int(count)):
            rng = random.Random(stable_seed(self.seed, split, "unknown", start_variant, index))
            values = [round4(rng.uniform(0.00, 0.08)) for _ in range(self.dimension)]
            rows.append(
                StrictCase(
                    case_id=f"{split}_{index:03d}",
                    split=split,
                    visible_state_items=[self._state_item(values)],
                    private_values=values,
                    judge_unknown=True,
                )
            )
        return rows

    def make_case(self, *, split: str, index: int, active_channels: dict[int, float], variant: int) -> StrictCase:
        rng = random.Random(stable_seed(self.seed, split, index, variant))
        values = [rng.uniform(0.00, 0.025) for _ in range(self.dimension)]
        if self.dimension > 6:
            values[6] = rng.uniform(0.00, 0.08)
        if self.dimension > 7:
            values[7] = rng.uniform(0.00, 0.08)
        for channel, value in active_channels.items():
            values[int(channel)] = float(value) + rng.uniform(-0.018, 0.018)
        rounded = [round4(value) for value in values]
        return StrictCase(
            case_id=f"{split}_{index:03d}",
            split=split,
            visible_state_items=[self._state_item(rounded)],
            private_values=rounded,
        )

    def judge(self, case: StrictCase, action_id: str) -> dict:
        if case.judge_unknown:
            rewarded = str(action_id) == HOLD_ACTION
            verdict = "held_unknown" if rewarded else "acted_on_unknown"
        else:
            rewarded = str(action_id) == self.private_best_action(case)
            verdict = "rewarded" if rewarded else "punished"
        return {
            "verdict": verdict,
            "reward": 1.0 if rewarded else 0.0,
            "punishment": 0.0 if rewarded else 1.0,
            "quality_tags": ["external_score_after_commit"],
            "answer_payload": None,
        }

    def perform_action(self, *, case: StrictCase, action_id: str) -> dict:
        return {
            "event_type": "external_actuator_event",
            "actuator_schema": "strict_compositional_feature_world/v1",
            "case_id": case.case_id,
            "executed_action_id": str(action_id),
            "execution_result": "committed",
            "environment_changed": True,
        }

    def private_best_action(self, case: StrictCase) -> str:
        scores = self._private_scores(list(case.private_values or []))
        return max(CHOICE_ACTIONS, key=lambda action: (scores[action], action))

    def _private_scores(self, values: list[float]) -> dict[str, float]:
        return {
            action: sum(float(values[channel]) for channel in channels)
            for action, channels in self._private_channels_by_action.items()
        }

    @staticmethod
    def inverted_feedback(feedback: dict) -> dict:
        return BlindButtonWorld.inverted_feedback(feedback)

    @staticmethod
    def random_feedback(*, rng: random.Random) -> dict:
        return BlindButtonWorld.random_feedback(rng=rng)

    @staticmethod
    def _state_item(values: list[float]) -> dict:
        return {
            "sa_label": "sensor::strict_compositional_features",
            "display_text": "strict compositional numeric features",
            "source_type": "strict_numeric_sensor",
            "family": "numeric_sensor",
            "real_energy": 1.0,
            "numeric_features": {f"feat.v{i}": round4(value) for i, value in enumerate(values)},
            "anchor_meta": {
                "schema_id": "strict_compositional_feature_sensor/v1",
                "numeric_only": True,
            },
        }
