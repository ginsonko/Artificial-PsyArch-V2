from __future__ import annotations

import math


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionParameterMemory:
    """
    Learns parameterized action styles without becoming concept learning.

    `ActionOutcomeMemory` answers "is this action generally good here?".
    This memory answers the narrower actuator question: "when this action was
    selected, which parameter pattern worked?". For visual gaze that means old
    center, new center, delta, bbox target, and observed feedback. The memory
    only returns a soft drive/pressure bias; normal AP action competition still
    decides whether the action happens.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        learning_rate: float = 0.20,
        decay_per_tick: float = 0.994,
        support_scale: float = 3.0,
        max_records_per_action: int = 24,
        max_drive_bias: float = 0.22,
    ) -> None:
        self.enabled = bool(enabled)
        self.learning_rate = _clamp(float(learning_rate), 0.001, 1.0)
        self.decay_per_tick = _clamp(float(decay_per_tick), 0.90, 1.0)
        self.support_scale = max(0.5, float(support_scale))
        self.max_records_per_action = max(4, int(max_records_per_action))
        self.max_drive_bias = max(0.0, float(max_drive_bias))
        self._records: dict[str, list[dict]] = {}
        self._last_tick = -1

    def advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(0, int(tick_index) - int(self._last_tick))
        if delta <= 0:
            return
        decay = self.decay_per_tick**delta
        for action_id in list(self._records):
            rows = []
            for row in self._records[action_id]:
                item = dict(row)
                item["support"] = float(item.get("support", 0.0) or 0.0) * decay
                item["utility"] = float(item.get("utility", 0.0) or 0.0) * decay
                if float(item.get("support", 0.0) or 0.0) >= 0.001:
                    rows.append(item)
            if rows:
                self._records[action_id] = rows[-self.max_records_per_action :]
            else:
                self._records.pop(action_id, None)
        self._last_tick = int(tick_index)

    def record(
        self,
        *,
        action_id: str,
        selected_action: dict,
        control_event: dict | None,
        observed_feedback: dict,
        tick_index: int | None = None,
    ) -> dict:
        action = str(action_id or "")
        if not self.enabled or not action:
            return self._empty_estimate(action)
        event = dict(control_event or {})
        params = dict((selected_action or {}).get("params", {}) or {})
        if self._is_parameterized_text_action(action, event, params):
            sample = self._sample_from_text_action_event(
                action_id=action,
                params=params,
                event=event,
                observed_feedback=observed_feedback,
                tick_index=tick_index,
            )
            rows = self._records.setdefault(action, [])
            nearest_index = self._nearest_record_index(rows, sample)
            if nearest_index is None:
                rows.append(sample)
            else:
                rows[nearest_index] = self._merge_record(rows[nearest_index], sample)
            rows.sort(key=lambda row: (-abs(float(row.get("utility", 0.0) or 0.0)) * float(row.get("support", 0.0) or 0.0), -float(row.get("support", 0.0) or 0.0)))
            self._records[action] = rows[: self.max_records_per_action]
            return self.estimate(action_id=action, proposed_params=params)
        if not self._is_parameterized_gaze_action(action, event, params):
            return self._empty_estimate(action)
        sample = self._sample_from_action_event(action_id=action, params=params, event=event, observed_feedback=observed_feedback, tick_index=tick_index)
        if not sample.get("has_delta", False):
            return self._empty_estimate(action)
        rows = self._records.setdefault(action, [])
        nearest_index = self._nearest_record_index(rows, sample)
        if nearest_index is None:
            rows.append(sample)
        else:
            rows[nearest_index] = self._merge_record(rows[nearest_index], sample)
        rows.sort(key=lambda row: (-abs(float(row.get("utility", 0.0) or 0.0)) * float(row.get("support", 0.0) or 0.0), -float(row.get("support", 0.0) or 0.0)))
        self._records[action] = rows[: self.max_records_per_action]
        return self.estimate(action_id=action, proposed_params=params, current_gaze=self._old_center_from_event(event))

    def estimate(self, *, action_id: str, proposed_params: dict | None = None, current_gaze: dict | None = None) -> dict:
        action = str(action_id or "")
        rows = list(self._records.get(action, []) or [])
        if not self.enabled or not rows:
            return self._empty_estimate(action)
        proposal = self._proposal_vector(proposed_params or {}, current_gaze or {})
        scored = []
        for row in rows:
            similarity = self._similarity(row, proposal)
            support = self._support_gate(float(row.get("support", 0.0) or 0.0))
            utility = float(row.get("utility", 0.0) or 0.0)
            score = similarity * support
            if score <= 0.0:
                continue
            scored.append((score, similarity, support, utility, row))
        if not scored:
            return self._empty_estimate(action)
        scored.sort(key=lambda item: (-item[0], -abs(item[3])))
        best_score, similarity, support, utility, row = scored[0]
        drive_bias = _clamp(utility * best_score * 0.55, -self.max_drive_bias, self.max_drive_bias)
        return {
            "schema_id": "action_parameter_estimate/v1",
            "method": "parameterized_action_style_memory",
            "action_id": action,
            "support": _round4(support),
            "similarity": _round4(similarity),
            "score": _round4(best_score),
            "utility": _round4(utility),
            "drive_bias": _round4(drive_bias),
            "pressure_bias": _round4(max(0.0, -drive_bias) * 0.65),
            "record_count": len(rows),
            "best_record": self._public_record(row),
        }

    def snapshot(self) -> dict:
        estimates = []
        for action_id in sorted(self._records):
            rows = [self._public_record(row) for row in self._records[action_id]]
            if rows:
                estimates.append({"action_id": action_id, "records": rows})
        return {
            "schema_id": "action_parameter_memory/v1",
            "enabled": bool(self.enabled),
            "policy": {
                "learning_rate": _round4(self.learning_rate),
                "decay_per_tick": _round4(self.decay_per_tick),
                "support_scale": _round4(self.support_scale),
                "max_records_per_action": self.max_records_per_action,
                "max_drive_bias": _round4(self.max_drive_bias),
            },
            "action_count": len(estimates),
            "estimates": estimates,
        }

    def _is_parameterized_gaze_action(self, action_id: str, event: dict, params: dict) -> bool:
        if action_id not in {"action::move_gaze_to", "action::nudge_gaze", "action::scan_visual_field", "action::hold_gaze"}:
            return False
        return bool(event) or "x" in params or "y" in params or "bbox_norm" in params or "dx" in params or "dy" in params

    def _is_parameterized_text_action(self, action_id: str, event: dict, params: dict) -> bool:
        if action_id not in {"action::text_insert", "action::text_delete", "action::text_replace"}:
            return False
        return bool(event) or "span" in params or "new_text" in params or "token" in params or "cursor" in params

    def _sample_from_action_event(self, *, action_id: str, params: dict, event: dict, observed_feedback: dict, tick_index: int | None) -> dict:
        old_center = self._center_pair(event, "old_center_x", "old_center_y", fallback=(0.5, 0.5))
        new_center = self._center_pair(event, "center_x", "center_y", fallback=self._target_pair(params, fallback=old_center))
        delta = [_round4(new_center[0] - old_center[0]), _round4(new_center[1] - old_center[1])]
        movement_distance = float(event.get("movement_distance", math.sqrt(delta[0] ** 2 + delta[1] ** 2)) or 0.0)
        feedback = dict(observed_feedback or {})
        reward = max(0.0, float(feedback.get("reward", 0.0) or 0.0))
        punishment = max(0.0, float(feedback.get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float(feedback.get("correctness", 0.0) or 0.0))
        confidence = _clamp(float(feedback.get("confidence", 0.0) or 0.0), 0.0, 1.0)
        utility = reward + correctness * 0.42 - punishment * 1.08
        support = _clamp((0.2 + confidence * 0.8) * (0.25 + abs(utility) + movement_distance * 0.6), 0.02, 1.0)
        return {
            "schema_id": "action_parameter_record/v1",
            "action_id": action_id,
            "tick_index": None if tick_index is None else int(tick_index),
            "target": str(event.get("target", params.get("target", "")) or ""),
            "reason": str(event.get("reason", params.get("reason", "")) or ""),
            "old_center_norm": [_round4(old_center[0]), _round4(old_center[1])],
            "new_center_norm": [_round4(new_center[0]), _round4(new_center[1])],
            "delta_norm": delta,
            "movement_distance": _round4(movement_distance),
            "bbox_norm": self._bbox(params, event),
            "utility": _round4(utility),
            "support": _round4(support),
            "event_count": 1,
            "last_feedback": {
                "reward": _round4(reward),
                "punishment": _round4(punishment),
                "correctness": _round4(correctness),
                "confidence": _round4(confidence),
                "utility": _round4(utility),
            },
            "has_delta": abs(delta[0]) > 0.0001 or abs(delta[1]) > 0.0001 or movement_distance > 0.0001,
        }

    def _sample_from_text_action_event(self, *, action_id: str, params: dict, event: dict, observed_feedback: dict, tick_index: int | None) -> dict:
        event_type = str(event.get("event_type", "") or "")
        if action_id == "action::text_insert" or event_type == "insert":
            kind = "text_insert"
            cursor = int(event.get("cursor_before", params.get("cursor", 0)) or 0)
            token = str(event.get("token", params.get("token", params.get("text", ""))) or "")
            span = (cursor, cursor)
            from_token = ""
            new_text = token
        elif action_id == "action::text_delete" or event_type == "delete":
            kind = "text_delete"
            span = self._text_span(event.get("span", params.get("span")))
            from_token = str(event.get("token", params.get("from_token", "")) or "")
            new_text = ""
        else:
            kind = "text_replace"
            span = self._text_span(event.get("span", params.get("span")))
            from_token = str(event.get("from_token", params.get("from_token", "")) or "")
            new_text = str(event.get("to_token", params.get("new_text", params.get("token", ""))) or "")
        expected = str(
            event.get("action_expected_token", "")
            or event.get("feedback_expected_token", "")
            or event.get("expected_token", "")
            or params.get("expected_token", params.get("candidate_token", new_text))
            or ""
        )
        conflict_index = int(event.get("target_index", params.get("conflict_index", params.get("cursor", span[0]))) or 0)
        feedback = dict(observed_feedback or {})
        reward = max(0.0, float(feedback.get("reward", 0.0) or 0.0))
        punishment = max(0.0, float(feedback.get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float(feedback.get("correctness", 0.0) or 0.0))
        confidence = _clamp(float(feedback.get("confidence", 0.0) or 0.0), 0.0, 1.0)
        utility = reward + correctness * 0.42 - punishment * 1.08
        span_len = max(0, int(span[1]) - int(span[0]))
        specificity = (
            0.18
            + (0.12 if new_text else 0.0)
            + (0.08 if from_token else 0.0)
            + min(0.12, max(1 if kind == "text_insert" else 0, span_len) * 0.04)
        )
        support = _clamp((0.2 + confidence * 0.8) * (specificity + abs(utility)), 0.02, 1.0)
        return {
            "schema_id": "action_parameter_record/v1",
            "parameter_kind": kind,
            "action_id": action_id,
            "tick_index": None if tick_index is None else int(tick_index),
            "target": str(new_text or expected or ""),
            "reason": str(event.get("reason", params.get("reason", "")) or ""),
            "span": list(span),
            "span_start": int(span[0]),
            "span_len": span_len,
            "conflict_index": int(conflict_index),
            "cursor": int(conflict_index if kind == "text_insert" else span[0]),
            "from_token": from_token,
            "new_text": new_text,
            "expected_token": expected,
            "utility": _round4(utility),
            "support": _round4(support),
            "event_count": 1,
            "last_feedback": {
                "reward": _round4(reward),
                "punishment": _round4(punishment),
                "correctness": _round4(correctness),
                "confidence": _round4(confidence),
                "utility": _round4(utility),
            },
        }

    def _merge_record(self, old: dict, sample: dict) -> dict:
        alpha = self.learning_rate
        merged = dict(old)
        for key in ("utility", "support", "movement_distance"):
            merged[key] = _round4(float(old.get(key, 0.0) or 0.0) * (1.0 - alpha) + float(sample.get(key, 0.0) or 0.0) * alpha)
        for key in ("old_center_norm", "new_center_norm", "delta_norm", "bbox_norm"):
            merged[key] = self._merge_vector(list(old.get(key, []) or []), list(sample.get(key, []) or []), alpha=alpha)
        old_kind = str(old.get("parameter_kind", "") or "")
        sample_kind = str(sample.get("parameter_kind", "") or "")
        if sample_kind.startswith("text_") or old_kind.startswith("text_"):
            for key in ("span_start", "span_len", "conflict_index"):
                merged[key] = _round4(float(old.get(key, 0.0) or 0.0) * (1.0 - alpha) + float(sample.get(key, 0.0) or 0.0) * alpha)
            if "cursor" in sample or "cursor" in old:
                merged["cursor"] = _round4(float(old.get("cursor", old.get("span_start", 0.0)) or 0.0) * (1.0 - alpha) + float(sample.get("cursor", sample.get("span_start", 0.0)) or 0.0) * alpha)
            for key in ("span", "from_token", "new_text", "expected_token"):
                merged[key] = sample.get(key, old.get(key))
            merged["parameter_kind"] = sample_kind or old_kind
        merged["target"] = str(sample.get("target", "") or old.get("target", "") or "")
        merged["reason"] = str(sample.get("reason", "") or old.get("reason", "") or "")
        merged["event_count"] = int(old.get("event_count", 0) or 0) + 1
        merged["tick_index"] = sample.get("tick_index", old.get("tick_index"))
        merged["last_feedback"] = dict(sample.get("last_feedback", {}) or {})
        return merged

    def _nearest_record_index(self, rows: list[dict], sample: dict) -> int | None:
        if not rows:
            return None
        best_index = None
        best_similarity = 0.0
        proposal = self._proposal_vector(sample, {})
        for idx, row in enumerate(rows):
            similarity = self._similarity(row, proposal)
            if similarity > best_similarity:
                best_index = idx
                best_similarity = similarity
        return best_index if best_similarity >= 0.72 else None

    def _proposal_vector(self, params: dict, current_gaze: dict) -> dict:
        text_kind = str(params.get("parameter_kind", "") or "")
        if text_kind.startswith("text_") or "span" in params or "new_text" in params or "token" in params or "conflict_index" in params or "cursor" in params:
            span = self._text_span(params.get("span"))
            cursor = int(params.get("cursor", params.get("conflict_index", span[0])) or 0)
            if text_kind == "text_insert" and "span" not in params:
                span = (cursor, cursor)
            return {
                "parameter_kind": text_kind or "text_replace",
                "span": list(span),
                "span_start": int(span[0]),
                "span_len": max(0, int(span[1]) - int(span[0])),
                "conflict_index": int(params.get("conflict_index", cursor if text_kind == "text_insert" else span[0]) or 0),
                "cursor": cursor,
                "from_token": str(params.get("from_token", "") or ""),
                "new_text": str(params.get("new_text", params.get("token", "")) or ""),
                "expected_token": str(
                    params.get("action_expected_token", "")
                    or params.get("feedback_expected_token", "")
                    or params.get("expected_token", params.get("candidate_token", params.get("new_text", params.get("token", ""))))
                    or ""
                ),
            }
        old_x = float(current_gaze.get("center_x", current_gaze.get("old_center_x", 0.5)) or 0.5)
        old_y = float(current_gaze.get("center_y", current_gaze.get("old_center_y", 0.5)) or 0.5)
        target = self._target_pair(params, fallback=(old_x, old_y))
        return {
            "target_xy": [_round4(target[0]), _round4(target[1])],
            "delta_norm": [_round4(target[0] - old_x), _round4(target[1] - old_y)],
            "bbox_norm": self._bbox(params, {}),
        }

    def _similarity(self, row: dict, proposal: dict) -> float:
        if str(row.get("parameter_kind", "") or "").startswith("text_") or str(proposal.get("parameter_kind", "") or "").startswith("text_"):
            return self._text_similarity(row, proposal)
        row_delta = list(row.get("delta_norm", []) or [])
        prop_delta = list(proposal.get("delta_norm", []) or [])
        delta_sim = self._vector_similarity(row_delta, prop_delta, scale=0.42)
        row_xy = list(row.get("new_center_norm", []) or [])
        prop_xy = list(proposal.get("target_xy", []) or [])
        xy_sim = self._vector_similarity(row_xy, prop_xy, scale=0.55)
        bbox_sim = self._vector_similarity(list(row.get("bbox_norm", []) or []), list(proposal.get("bbox_norm", []) or []), scale=0.70)
        if not proposal.get("bbox_norm"):
            bbox_sim = 0.0
        return _clamp(delta_sim * 0.58 + xy_sim * 0.30 + bbox_sim * 0.12, 0.0, 1.0)

    def _text_similarity(self, row: dict, proposal: dict) -> float:
        row_start = float(row.get("span_start", row.get("conflict_index", 0)) or 0.0)
        prop_start = float(proposal.get("span_start", proposal.get("conflict_index", 0)) or 0.0)
        row_len = float(row.get("span_len", 1) or 1.0)
        prop_len = float(proposal.get("span_len", 1) or 1.0)
        row_kind = str(row.get("parameter_kind", "") or "")
        prop_kind = str(proposal.get("parameter_kind", "") or "")
        kind_sim = 1.0 if row_kind and prop_kind and row_kind == prop_kind else (0.35 if row_kind.startswith("text_") and prop_kind.startswith("text_") else 0.0)
        position_sim = math.exp(-abs(row_start - prop_start) / 3.0)
        length_sim = math.exp(-abs(row_len - prop_len) / 2.0)
        row_new = str(row.get("new_text", row.get("expected_token", "")) or "")
        prop_new = str(proposal.get("new_text", proposal.get("expected_token", "")) or "")
        row_expected = str(row.get("expected_token", row_new) or "")
        prop_expected = str(proposal.get("expected_token", prop_new) or "")
        token_sim = 0.0
        if row_new and prop_new and row_new == prop_new:
            token_sim = 1.0
        elif row_expected and prop_expected and row_expected == prop_expected:
            token_sim = 0.82
        elif row_new and prop_expected and row_new == prop_expected:
            token_sim = 0.74
        elif prop_new:
            token_sim = 0.18
        from_sim = 0.0
        row_from = str(row.get("from_token", "") or "")
        prop_from = str(proposal.get("from_token", "") or "")
        if row_from and prop_from and row_from == prop_from:
            from_sim = 1.0
        elif not prop_from:
            from_sim = 0.18
        return _clamp(kind_sim * 0.16 + position_sim * 0.24 + length_sim * 0.10 + token_sim * 0.38 + from_sim * 0.12, 0.0, 1.0)

    def _vector_similarity(self, left: list[float], right: list[float], *, scale: float) -> float:
        if not left or not right:
            return 0.0
        count = min(len(left), len(right))
        distance = math.sqrt(sum((float(left[idx]) - float(right[idx])) ** 2 for idx in range(count)))
        return math.exp(-distance / max(0.001, float(scale)))

    def _support_gate(self, support: float) -> float:
        return _clamp(float(support) / (float(support) + self.support_scale), 0.0, 1.0)

    def _center_pair(self, event: dict, x_key: str, y_key: str, *, fallback: tuple[float, float]) -> tuple[float, float]:
        return (
            _clamp(float(event.get(x_key, fallback[0]) or fallback[0]), 0.0, 1.0),
            _clamp(float(event.get(y_key, fallback[1]) or fallback[1]), 0.0, 1.0),
        )

    def _old_center_from_event(self, event: dict) -> dict:
        return {
            "center_x": float(event.get("old_center_x", 0.5) or 0.5),
            "center_y": float(event.get("old_center_y", 0.5) or 0.5),
        }

    def _target_pair(self, params: dict, *, fallback: tuple[float, float]) -> tuple[float, float]:
        if "x" in params or "y" in params:
            return (
                _clamp(float(params.get("x", fallback[0]) or fallback[0]), 0.0, 1.0),
                _clamp(float(params.get("y", fallback[1]) or fallback[1]), 0.0, 1.0),
            )
        bbox = list(params.get("bbox_norm", []) or [])
        if len(bbox) >= 2:
            return (_clamp(float(bbox[0] or fallback[0]), 0.0, 1.0), _clamp(float(bbox[1] or fallback[1]), 0.0, 1.0))
        dx = params.get("dx")
        dy = params.get("dy")
        if dx is not None or dy is not None:
            return (
                _clamp(fallback[0] + float(dx or 0.0), 0.0, 1.0),
                _clamp(fallback[1] + float(dy or 0.0), 0.0, 1.0),
            )
        return fallback

    def _bbox(self, params: dict, event: dict) -> list[float]:
        bbox = list(event.get("bbox_norm", []) or params.get("bbox_norm", []) or [])
        return [_round4(_clamp(float(value or 0.0), 0.0, 1.0)) for value in bbox[:4]]

    def _text_span(self, span) -> tuple[int, int]:
        if isinstance(span, dict):
            start = span.get("start", span.get("from", span.get("begin", 0)))
            end = span.get("end", span.get("to", span.get("stop", None)))
            try:
                left = int(start or 0)
                right = int(end if end is not None else left + 1)
            except (TypeError, ValueError):
                return (0, 1)
            return (max(0, left), max(max(0, left), right))
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            try:
                left = int(span[0])
                right = int(span[1])
            except (TypeError, ValueError):
                return (0, 1)
            return (max(0, left), max(max(0, left), right))
        return (0, 1)

    def _merge_vector(self, old: list[float], new: list[float], *, alpha: float) -> list[float]:
        if not old:
            return [_round4(float(value or 0.0)) for value in new]
        if not new:
            return [_round4(float(value or 0.0)) for value in old]
        count = min(len(old), len(new))
        rows = [_round4(float(old[idx] or 0.0) * (1.0 - alpha) + float(new[idx] or 0.0) * alpha) for idx in range(count)]
        rows.extend([_round4(float(value or 0.0)) for value in old[count:]])
        return rows

    def _public_record(self, row: dict) -> dict:
        return {
            "schema_id": "action_parameter_record_view/v1",
            "action_id": str(row.get("action_id", "") or ""),
            "parameter_kind": str(row.get("parameter_kind", "") or ""),
            "target": str(row.get("target", "") or ""),
            "reason": str(row.get("reason", "") or ""),
            "old_center_norm": list(row.get("old_center_norm", []) or []),
            "new_center_norm": list(row.get("new_center_norm", []) or []),
            "delta_norm": list(row.get("delta_norm", []) or []),
            "movement_distance": _round4(float(row.get("movement_distance", 0.0) or 0.0)),
            "bbox_norm": list(row.get("bbox_norm", []) or []),
            "span": list(row.get("span", []) or []),
            "conflict_index": row.get("conflict_index", None),
            "cursor": row.get("cursor", None),
            "from_token": str(row.get("from_token", "") or ""),
            "new_text": str(row.get("new_text", "") or ""),
            "expected_token": str(row.get("expected_token", "") or ""),
            "utility": _round4(float(row.get("utility", 0.0) or 0.0)),
            "support": _round4(float(row.get("support", 0.0) or 0.0)),
            "event_count": int(row.get("event_count", 0) or 0),
            "last_feedback": dict(row.get("last_feedback", {}) or {}),
        }

    def _empty_estimate(self, action_id: str) -> dict:
        return {
            "schema_id": "action_parameter_estimate/v1",
            "method": "no_parameterized_action_evidence",
            "action_id": str(action_id or ""),
            "support": 0.0,
            "similarity": 0.0,
            "score": 0.0,
            "utility": 0.0,
            "drive_bias": 0.0,
            "pressure_bias": 0.0,
            "record_count": len(self._records.get(str(action_id or ""), []) or []),
            "best_record": {},
        }
