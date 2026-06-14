from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SECRET_RULE_TAG = "SECRET_RULE"
SECRET_TRUTH_TAG = "SECRET_TRUTH"


def round4(value: float) -> float:
    return round(float(value), 4)


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def ratio(values: list[bool]) -> float:
    return sum(1 for value in values if value) / max(1, len(values))


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_json_strings(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def action_feedback_item(*, action_id: str, feedback: dict, tick_index: int, case_id: str, source: str) -> dict:
    reward = float(feedback.get("reward", 0.0) or 0.0)
    punishment = float(feedback.get("punishment", 0.0) or 0.0)
    verdict = str(feedback.get("verdict", "") or "")
    return {
        "sa_label": f"action_feedback::{action_id}::{verdict}",
        "display_text": f"{action_id} {verdict}",
        "family": "action_feedback",
        "source_type": "action_feedback",
        "real_energy": 1.0 + reward,
        "virtual_energy": punishment * 0.15,
        "anchor_meta": {
            "schema_id": "strict_action_feedback/v1",
            "action_id": str(action_id),
            "case_id": str(case_id),
            "reward": round4(reward),
            "punishment": round4(punishment),
            "verdict": verdict,
            "source": str(source),
            "tick_index": int(tick_index),
        },
    }


def learner_feedback(feedback: dict) -> dict:
    """Return the only feedback payload allowed to reach the learner."""

    return {
        "schema_id": "strict_reward_punishment_feedback/v1",
        "reward": round4(float(feedback.get("reward", 0.0) or 0.0)),
        "punishment": round4(float(feedback.get("punishment", 0.0) or 0.0)),
        "verdict": str(feedback.get("verdict", "") or ""),
        "quality_tags": [
            str(tag or "")
            for tag in list(feedback.get("quality_tags", []) or [])
            if str(tag or "")
        ],
    }


def action_item(*, action_id: str, tick_index: int, reason: str) -> dict:
    return {
        "sa_label": str(action_id),
        "display_text": str(action_id),
        "family": "action",
        "source_type": "action_selection",
        "real_energy": 1.0,
        "anchor_meta": {
            "schema_id": "strict_selected_action/v1",
            "action_id": str(action_id),
            "reason": str(reason),
            "tick_index": int(tick_index),
        },
    }
