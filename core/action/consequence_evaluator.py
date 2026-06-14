from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionConsequenceEvaluator:
    """
    White-box action consequence reader.

    This module does not define Bn/Cn. It consumes the already-built Bn/Cn
    rows and looks into bounded successor snapshots for action-feedback SAs.
    That gives the planner an experience-based estimate:

        similar state -> successor field C* -> action feedback -> drive modulation
    """

    DEFAULT_ACTION_IDS = (
        "action::focus_anchor",
        "action::continue_focus",
        "action::release_focus",
        "action::diverge_attention",
        "action::inspect_residual",
        "action::scan_visual_field",
        "action::move_gaze_to",
        "action::widen_visual_focus",
        "action::widen_audio_band",
        "action::replay_recent_context",
        "action::recall_by_timefelt",
        "action::recall_by_expectation",
        "action::stabilize_prediction",
        "action::text_reread",
        "action::text_insert",
        "action::text_replace",
        "action::text_delete",
        "action::text_commit",
        "action::wait",
        "action::avoid",
    )

    def __init__(
        self,
        *,
        max_successor_rows: int = 12,
        max_evidence_per_action: int = 8,
        max_horizon: int = 3,
        branching: int = 3,
        path_decay: float = 0.72,
    ) -> None:
        self.max_successor_rows = max(1, int(max_successor_rows))
        self.max_evidence_per_action = max(1, int(max_evidence_per_action))
        self.max_horizon = max(1, int(max_horizon))
        self.branching = max(1, int(branching))
        self.path_decay = _clamp(float(path_decay), 0.1, 1.0)

    def evaluate(
        self,
        *,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        snapshot_lookup,
        successor_lookup=None,
        action_ids: list[str] | None = None,
        current_tick: int | None = None,
    ) -> dict:
        wanted = [str(action_id or "") for action_id in (action_ids or self.DEFAULT_ACTION_IDS) if str(action_id or "")]
        estimates = {action_id: self._empty_estimate(action_id) for action_id in wanted}
        direct_context = self._successor_context_rows(
            fast_bn=fast_bn,
            fast_cn=fast_cn,
            slow_bn=slow_bn,
            slow_cn=slow_cn,
            snapshot_lookup=snapshot_lookup,
        )
        source_context = self._expand_consequence_context(
            direct_context,
            snapshot_lookup=snapshot_lookup,
            successor_lookup=successor_lookup,
            current_tick=current_tick,
        )

        evidence_by_action: dict[str, list[dict]] = {action_id: [] for action_id in wanted}
        wanted_set = set(wanted)
        for row in source_context:
            snapshot = dict(row.get("snapshot", {}) or {})
            score = float(row.get("score", 0.0) or 0.0)
            branch = str(row.get("branch", "") or "")
            feedback_items = snapshot.get("action_feedback_items", None)
            if not isinstance(feedback_items, list):
                feedback_items = snapshot.get("items", []) or []
            for item in feedback_items:
                action_id = self._action_id_from_feedback_item(item)
                if not action_id or action_id not in wanted_set:
                    continue
                feedback = self._observed_feedback_from_item(item)
                if not feedback:
                    continue
                causal_window = self._causal_window_from_item(item)
                evidence_by_action[action_id].append(
                    {
                        "action_id": action_id,
                        "source_memory_id": str(row.get("source_memory_id", "") or ""),
                        "successor_memory_id": str(row.get("successor_memory_id", "") or ""),
                        "branch": branch,
                        "depth": int(row.get("depth", 1) or 1),
                        "path": list(row.get("path", []) or []),
                        "source_score": _round4(score),
                        "temporal_applicability": _round4(float(row.get("temporal_applicability", 1.0) or 1.0)),
                        "temporal_age_ticks": row.get("temporal_age_ticks"),
                        "temporal_applicability_phase": str(row.get("temporal_applicability_phase", "") or ""),
                        "observed_feedback": feedback,
                        "causal_window": causal_window,
                        "sa_label": str(item.get("sa_label", "") or ""),
                    }
                )

        for action_id, evidence_rows in evidence_by_action.items():
            estimates[action_id] = self._estimate_from_evidence(action_id, evidence_rows)

        supported = [row for row in estimates.values() if float(row.get("support", 0.0) or 0.0) > 0.0]
        supported.sort(key=lambda item: (-float(item.get("support", 0.0) or 0.0), str(item.get("action_id", "") or "")))
        return {
            "schema_id": "action_consequence_trace/v1",
            "source": "successor_action_feedback",
            "path_policy": {
                "max_horizon": int(self.max_horizon),
                "branching": int(self.branching),
                "path_decay": _round4(self.path_decay),
                "max_successor_rows": int(self.max_successor_rows),
            },
            "direct_successor_context_count": len(direct_context),
            "successor_context_count": len(source_context),
            "supported_action_count": len(supported),
            "action_estimates": estimates,
            "top_supported_actions": supported[:4],
        }

    def _successor_context_rows(
        self,
        *,
        fast_bn: list[dict],
        fast_cn: list[dict],
        slow_bn: list[dict],
        slow_cn: list[dict],
        snapshot_lookup,
    ) -> list[dict]:
        bn_score_by_id: dict[str, float] = {}
        for row in list(fast_bn or []) + list(slow_bn or []):
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id:
                continue
            bn_score_by_id[memory_id] = max(
                float(bn_score_by_id.get(memory_id, 0.0) or 0.0),
                float(row.get("score", 0.0) or 0.0),
            )

        rows = []
        seen = set()
        for branch_name, cn_rows in (("fast", fast_cn), ("slow", slow_cn)):
            for cn_row in cn_rows or []:
                source_id = str(cn_row.get("source_memory_id", "") or "")
                successor_id = str(cn_row.get("successor_memory_id", "") or "")
                if not successor_id:
                    continue
                key = (branch_name, source_id, successor_id)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    snapshot = snapshot_lookup(successor_id)
                except TypeError:
                    snapshot = None
                if not snapshot:
                    continue
                cn_score = float(cn_row.get("score", 0.0) or 0.0)
                bn_score = float(bn_score_by_id.get(source_id, 0.0) or 0.0)
                score = max(0.01, cn_score) * max(0.1, min(1.0, bn_score if bn_score > 0 else 0.35))
                rows.append(
                    {
                        "branch": branch_name,
                        "source_memory_id": source_id,
                        "successor_memory_id": successor_id,
                        "score": _round4(score),
                        "depth": 1,
                        "path": [source_id, successor_id] if source_id else [successor_id],
                        "snapshot": dict(snapshot),
                        "temporal_applicability": _round4(float(cn_row.get("temporal_applicability", 1.0) or 1.0)),
                        "temporal_age_ticks": cn_row.get("temporal_age_ticks"),
                        "temporal_applicability_phase": str(cn_row.get("temporal_applicability_phase", "") or ""),
                    }
                )

        rows.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                str(item.get("branch", "") or ""),
                str(item.get("successor_memory_id", "") or ""),
            )
        )
        return rows[: self.max_successor_rows]

    def _expand_consequence_context(self, direct_context: list[dict], *, snapshot_lookup, successor_lookup, current_tick: int | None = None) -> list[dict]:
        if not callable(successor_lookup) or self.max_horizon <= 1:
            return list(direct_context)[: self.max_successor_rows]
        rows: list[dict] = []
        queue: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for row in direct_context:
            current = dict(row)
            current["depth"] = max(1, int(current.get("depth", 1) or 1))
            current["path"] = list(current.get("path", []) or [current.get("successor_memory_id", "")])
            rows.append(current)
            queue.append(current)
            seen.add((str(current.get("successor_memory_id", "") or ""), int(current.get("depth", 1) or 1)))

        cursor = 0
        while cursor < len(queue) and len(rows) < self.max_successor_rows:
            row = queue[cursor]
            cursor += 1
            depth = int(row.get("depth", 1) or 1)
            if depth >= self.max_horizon:
                continue
            source_id = str(row.get("successor_memory_id", "") or "")
            if not source_id:
                continue
            try:
                next_rows = successor_lookup(source_id, memory_kind="state", top_k=self.branching, current_tick=current_tick)
            except TypeError:
                next_rows = successor_lookup(source_id)
            for link in (next_rows or [])[: self.branching]:
                successor_id = str((link or {}).get("successor_memory_id", "") or "")
                if not successor_id:
                    continue
                next_depth = depth + 1
                key = (successor_id, next_depth)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    snapshot = snapshot_lookup(successor_id)
                except TypeError:
                    snapshot = None
                if not snapshot:
                    continue
                score = float(row.get("score", 0.0) or 0.0) * max(0.01, float((link or {}).get("score", 1.0) or 1.0)) * self.path_decay
                path = list(row.get("path", []) or [])
                path.append(successor_id)
                expanded = {
                    "branch": f"{row.get('branch', '')}:h{next_depth}",
                    "source_memory_id": str(row.get("source_memory_id", "") or ""),
                    "successor_memory_id": successor_id,
                    "score": _round4(score),
                    "depth": next_depth,
                    "path": path[-max(2, self.max_horizon + 1) :],
                    "snapshot": dict(snapshot),
                    "temporal_applicability": _round4(float((link or {}).get("temporal_applicability", 1.0) or 1.0)),
                    "temporal_age_ticks": (link or {}).get("temporal_age_ticks"),
                    "temporal_applicability_phase": str((link or {}).get("temporal_applicability_phase", "") or ""),
                }
                rows.append(expanded)
                queue.append(expanded)
                if len(rows) >= self.max_successor_rows:
                    break
        rows.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                int(item.get("depth", 1) or 1),
                str(item.get("successor_memory_id", "") or ""),
            )
        )
        return rows[: self.max_successor_rows]

    def _action_id_from_feedback_item(self, item: dict) -> str:
        if not isinstance(item, dict):
            return ""
        label = str(item.get("sa_label", "") or "")
        family = str(item.get("family", "") or "")
        source_type = str(item.get("source_type", "") or "")
        if not label.startswith("action_feedback::") and family != "action_feedback" and source_type != "action_feedback":
            return ""
        anchor_meta = dict(item.get("anchor_meta", {}) or {})
        action_id = str(anchor_meta.get("action_id", "") or "")
        if action_id.startswith("action::"):
            return action_id
        if label.startswith("action_feedback::"):
            action_name = label.split("::", 1)[-1]
            return f"action::{action_name}" if action_name else ""
        return ""

    def _observed_feedback_from_item(self, item: dict) -> dict:
        anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item, dict) else {}
        feedback = dict(anchor_meta.get("observed_feedback", {}) or {})
        if not feedback:
            real_energy = float((item or {}).get("real_energy", 0.0) or 0.0)
            if real_energy <= 0.0:
                return {}
            feedback = {"reward": real_energy, "punishment": 0.0, "correctness": real_energy * 0.35, "confidence": 0.18}
        return {
            "reward": _round4(max(0.0, float(feedback.get("reward", 0.0) or 0.0))),
            "punishment": _round4(max(0.0, float(feedback.get("punishment", 0.0) or 0.0))),
            "correctness": _round4(max(0.0, float(feedback.get("correctness", 0.0) or 0.0))),
            "confidence": _round4(_clamp(float(feedback.get("confidence", 0.0) or 0.0), 0.0, 1.0)),
        }

    def _causal_window_from_item(self, item: dict) -> dict:
        anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item, dict) else {}
        window = dict(anchor_meta.get("causal_window", {}) or {})
        if window.get("schema_id") != "action_causal_window/v1":
            return {}
        text_output = dict(window.get("text_output", {}) or {})
        return {
            "schema_id": "action_causal_window/v1",
            "tick_index": int(window.get("tick_index", -1) or -1),
            "action_ids": list(window.get("action_ids", []) or [])[:4],
            "entered_labels": list(window.get("entered_labels", []) or [])[:6],
            "control_labels": list(window.get("control_labels", []) or [])[:6],
            "text_output": {
                "visible_text": str(text_output.get("visible_text", "") or ""),
                "expected_token": str(text_output.get("expected_token", "") or ""),
                "revision_detected": bool(text_output.get("revision_detected", False)),
                "output_item_labels": list(text_output.get("output_item_labels", []) or [])[:6],
            },
        }

    def _estimate_from_evidence(self, action_id: str, evidence_rows: list[dict]) -> dict:
        if not evidence_rows:
            return self._empty_estimate(action_id)
        rows = list(evidence_rows)[: self.max_evidence_per_action]
        weight_sum = 0.0
        totals = {"reward": 0.0, "punishment": 0.0, "correctness": 0.0, "confidence": 0.0}
        source_ids: list[str] = []
        evidence_paths: list[dict] = []
        outcome_windows: list[dict] = []
        max_depth = 0
        for row in rows:
            feedback = dict(row.get("observed_feedback", {}) or {})
            weight = max(0.05, float(row.get("source_score", 0.0) or 0.0))
            depth = max(1, int(row.get("depth", 1) or 1))
            max_depth = max(max_depth, depth)
            conf = _clamp(float(feedback.get("confidence", 0.0) or 0.0), 0.0, 1.0)
            weight *= 0.55 + conf * 0.45
            weight_sum += weight
            for key in totals:
                totals[key] += float(feedback.get(key, 0.0) or 0.0) * weight
            successor_id = str(row.get("successor_memory_id", "") or "")
            if successor_id and successor_id not in source_ids:
                source_ids.append(successor_id)
            if len(evidence_paths) < self.max_evidence_per_action:
                evidence_paths.append(
                    {
                        "successor_memory_id": successor_id,
                        "depth": depth,
                        "source_score": _round4(float(row.get("source_score", 0.0) or 0.0)),
                        "temporal_applicability": _round4(float(row.get("temporal_applicability", 1.0) or 1.0)),
                        "temporal_age_ticks": row.get("temporal_age_ticks"),
                        "temporal_applicability_phase": str(row.get("temporal_applicability_phase", "") or ""),
                        "path": list(row.get("path", []) or []),
                    }
                )
            causal_window = dict(row.get("causal_window", {}) or {})
            if causal_window and len(outcome_windows) < self.max_evidence_per_action:
                outcome_windows.append(
                    {
                        "tick_index": int(causal_window.get("tick_index", -1) or -1),
                        "action_ids": list(causal_window.get("action_ids", []) or [])[:4],
                        "entered_labels": list(causal_window.get("entered_labels", []) or [])[:6],
                        "text_output": dict(causal_window.get("text_output", {}) or {}),
                    }
                )
        if weight_sum <= 0.0:
            return self._empty_estimate(action_id)
        reward = totals["reward"] / weight_sum
        punishment = totals["punishment"] / weight_sum
        correctness = totals["correctness"] / weight_sum
        confidence = totals["confidence"] / weight_sum
        pressure = max(0.0, punishment * 0.72 - reward * 0.18)
        support = _clamp((len(rows) / float(self.max_evidence_per_action)) * 0.65 + min(0.35, weight_sum * 0.08), 0.0, 1.0)
        return {
            "action_id": action_id,
            "support": _round4(support),
            "evidence_count": len(rows),
            "weighted_support": _round4(weight_sum),
            "reward": _round4(reward),
            "punishment": _round4(punishment),
            "correctness": _round4(correctness),
            "pressure": _round4(pressure),
            "confidence": _round4(confidence),
            "source_memory_ids": source_ids[: self.max_evidence_per_action],
            "max_depth": int(max_depth),
            "evidence_paths": evidence_paths,
            "outcome_window_count": len(outcome_windows),
            "outcome_windows": outcome_windows,
            "method": "successor_action_feedback",
        }

    def _empty_estimate(self, action_id: str) -> dict:
        return {
            "action_id": str(action_id or ""),
            "support": 0.0,
            "evidence_count": 0,
            "weighted_support": 0.0,
            "reward": 0.0,
            "punishment": 0.0,
            "correctness": 0.0,
            "pressure": 0.0,
            "confidence": 0.0,
            "source_memory_ids": [],
            "max_depth": 0,
            "evidence_paths": [],
            "outcome_window_count": 0,
            "outcome_windows": [],
            "method": "no_successor_feedback",
        }
