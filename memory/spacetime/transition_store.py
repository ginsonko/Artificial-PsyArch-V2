from __future__ import annotations

from collections import defaultdict


def _round4(value: float) -> float:
    return round(float(value), 4)


PREDICTION_META_KEYS = {
    "schema_id",
    "event_type",
    "current_glyph_index",
    "current_glyph_role",
    "same_tick_binding_role",
    "prediction_payload_priority",
    "process_anchor_role",
    "visible_length",
    "cursor_index",
    "cursor",
    "last_visible_token",
    "operation",
    "conflict_kind",
    "span",
    "support",
    "task_id",
    "paradigm_id",
    "region_id",
    "readout_semantic_role",
    "readout_pattern_id",
    "semantic_frame_role",
    "dynamic_slot_role",
    "slot_role",
    "previous_prefix",
    "token",
    "candidate_token",
    "expected_token",
    "source",
    "source_event_type",
    "cursor_before",
    "cursor_after",
    "visible_text_before",
    "visible_text_after",
    "action_param_reason",
    "action_id",
    "parameter_kind",
    "self_generated",
    "readout_expected_token",
    "feedback_outcome",
    "feedback_reward",
    "feedback_punishment",
    "feedback_correctness",
    "reward_value",
    "punishment_value",
}


class TransitionStore:
    def __init__(self) -> None:
        self._next_by_kind: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._items_by_id: dict[str, dict] = {}

    def register_snapshot(self, snapshot: dict) -> None:
        memory_id = str(snapshot.get("memory_id", "") or "")
        if not memory_id:
            return
        self._items_by_id[memory_id] = snapshot

    def link_successor(self, memory_kind: str, source_memory_id: str, successor_memory_id: str) -> None:
        kind = str(memory_kind or "")
        source_id = str(source_memory_id or "")
        successor_id = str(successor_memory_id or "")
        if not kind or not source_id or not successor_id:
            return
        bucket = self._next_by_kind[kind][source_id]
        if successor_id in bucket:
            bucket.remove(successor_id)
        bucket.append(successor_id)
        if len(bucket) > 8:
            del bucket[0 : len(bucket) - 8]

    def remove_snapshot(self, memory_kind: str, memory_id: str) -> None:
        """
        Remove a snapshot and best-effort prune successor edges.

        Design constraints:
        - We must support bounded eviction without scanning the whole graph.
        - We therefore remove the node payload immediately.
        - We prune outgoing edges from this node and light-prune incoming edges
          by scanning only the kind-local adjacency map (bounded by kind size).

        NOTE:
        This pruning is not on the tick hot path in the expected configuration
        because eviction is rare (only when per-kind snapshot cap is exceeded).
        """

        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id:
            return
        self._items_by_id.pop(clean_id, None)
        # Drop outgoing edges.
        self._next_by_kind.get(kind, {}).pop(clean_id, None)
        # Light-prune incoming edges for this kind.
        bucket = self._next_by_kind.get(kind, {})
        for source_id, successor_ids in list(bucket.items()):
            if not successor_ids or clean_id not in successor_ids:
                continue
            pruned = [sid for sid in successor_ids if str(sid) != clean_id]
            bucket[source_id] = pruned

    def successors(
        self,
        memory_kind: str,
        source_memory_id: str,
        *,
        top_k: int,
        prediction_energy_scale: float,
        lag_shaping_enabled: bool = True,
    ) -> list[dict]:
        kind = str(memory_kind or "")
        source_id = str(source_memory_id or "")
        rows = []
        source_snapshot = self._items_by_id.get(source_id, {})
        try:
            source_tick = int(source_snapshot.get("tick_index", 0) or 0)
        except (TypeError, ValueError):
            source_tick = 0
        for successor_id in self._next_by_kind.get(kind, {}).get(source_id, [])[: max(1, int(top_k))]:
            snapshot = self._items_by_id.get(successor_id)
            if not snapshot:
                continue
            try:
                successor_tick = int(snapshot.get("tick_index", source_tick + 1) or (source_tick + 1))
            except (TypeError, ValueError):
                successor_tick = source_tick + 1
            lag = max(1, successor_tick - source_tick)
            lag_kernel = self._lag_kernel(lag) if lag_shaping_enabled else 1.0
            predicted_items = []
            # P1-K-4 theory alignment:
            # Cn is a successor readout of the same all-SA state field. Action,
            # feeling, feedback, and control labels are valid predicted objects,
            # not derived channels to be excluded from the main AP experience
            # field. `prediction_payload_items` remains the preferred bounded C*
            # payload; `state_field_items` is the fallback main recognition view.
            source_items = snapshot.get("prediction_payload_items", None)
            if not isinstance(source_items, list) or not source_items:
                source_items = snapshot.get("state_field_items", None)
            if not isinstance(source_items, list) or not source_items:
                source_items = snapshot.get("core_items", None)
            if not isinstance(source_items, list) or not source_items:
                source_items = snapshot.get("items", []) or []

            # Successor fallback policy: keep it bounded and keep current-tick
            # external evidence early, but do not exclude all-SA intuition items.
            def _is_external_evidence(row: dict) -> bool:
                src = str(row.get("source_type", "") or "")
                if src == "external_text":
                    return True
                if src.startswith("vision_bridge"):
                    return True
                if src.startswith("audio_bridge"):
                    return True
                return False

            if snapshot.get("prediction_payload_items"):
                ordered = list(source_items)
            else:
                successor_tick = snapshot.get("tick_index")
                external_now = []
                action_now = []
                rest = []
                for row in list(source_items):
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get("sa_label", "") or "")
                    family = str(row.get("family", "") or "")
                    source_type = str(row.get("source_type", "") or "")
                    if label.startswith("action::") or family == "action" or source_type == "action_selection":
                        action_now.append(row)
                        continue
                    # Use `last_seen_tick` to detect what was actually observed at the successor tick.
                    # `tick_index` on the row is the snapshot tick, not the observation tick (it can
                    # be stamped uniformly at write time). `last_seen_tick` is the correct signal.
                    if successor_tick is not None and int(row.get("last_seen_tick", -999999) or -999999) == int(successor_tick) and _is_external_evidence(row):
                        external_now.append(row)
                    else:
                        rest.append(row)

                ordered = list(external_now) + list(action_now[:4]) + list(rest)
            for item in ordered[: max(1, int(top_k))]:
                label = str(item.get("sa_label", "") or "")
                family = str(item.get("family", "predicted") or "predicted")
                source_type = str(item.get("source_type", "") or "")
                is_action_feedback = label.startswith("action_feedback::") or family == "action_feedback" or source_type == "action_feedback"
                is_action = label.startswith("action::") or family == "action" or source_type == "action_selection"
                source_energy = float(item.get("real_energy", 0.0) or 0.0)
                if is_action_feedback or label.startswith(("signal::reward", "signal::punishment", "reward::", "punishment::", "rwd::", "pun::")):
                    source_energy = max(source_energy, float(item.get("virtual_energy", 0.0) or 0.0))
                predicted = {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", "") or ""),
                    "family": family,
                    "source_type": "predicted",
                    "virtual_energy": _round4(source_energy * float(prediction_energy_scale) * lag_kernel),
                }
                prediction_meta = self._prediction_anchor_meta(item)
                if prediction_meta:
                    predicted["anchor_meta"] = prediction_meta
                    predicted["anchor_meta"]["successor_lag_ticks"] = int(lag)
                    predicted["anchor_meta"]["successor_lag_kernel"] = _round4(lag_kernel)
                if is_action:
                    predicted["source_type"] = "predicted_action"
                    meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
                    predicted["anchor_meta"] = {
                        **dict(predicted.get("anchor_meta", {}) or {}),
                        "schema_id": "predicted_action_tendency/v1",
                        "action_id": label,
                        "prediction_role": "successor_action_sa",
                        "source_family": family,
                        "source_type": source_type,
                        "source_tick_index": meta.get("tick_index", item.get("tick_index")),
                        "successor_lag_ticks": int(lag),
                        "successor_lag_kernel": _round4(lag_kernel),
                        "learning_boundary": "action_prediction_shapes_drive_not_concept_embedding",
                    }
                if is_action_feedback:
                    meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
                    action_meta = dict(predicted.get("anchor_meta", {}) or {})
                    action_meta.update({
                        "schema_id": "predicted_action_feedback_outcome/v1",
                        "action_id": str(meta.get("action_id", "") or ""),
                        "observed_feedback": dict(meta.get("observed_feedback", {}) or {}),
                        "predicted_outcome": dict(meta.get("predicted_outcome", {}) or {}),
                        "feedback_energy_semantics": dict(meta.get("feedback_energy_semantics", {}) or {}),
                        "successor_lag_ticks": int(lag),
                        "successor_lag_kernel": _round4(lag_kernel),
                    })
                    predicted["anchor_meta"] = action_meta
                predicted_items.append(predicted)
            rows.append(
                {
                    "source_memory_id": source_id,
                    "successor_memory_id": successor_id,
                    "score": _round4(1.0 * lag_kernel),
                    "successor_lag_ticks": int(lag),
                    "successor_lag_kernel": _round4(lag_kernel),
                    "predicted_items": predicted_items,
                }
            )
        return rows

    def _lag_kernel(self, lag: int) -> float:
        clean_lag = max(1, int(lag))
        if clean_lag <= 1:
            return 1.0
        if clean_lag == 2:
            return 0.42
        return max(0.08, 0.42 * (0.64 ** (clean_lag - 2)))

    def _prediction_anchor_meta(self, item: dict) -> dict:
        meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        if not meta:
            return {}
        clean = {
            key: value
            for key, value in meta.items()
            if key in PREDICTION_META_KEYS and value is not None
        }
        if not clean:
            return {}
        clean["schema_id"] = str(clean.get("schema_id", "") or "predicted_payload_process_anchor/v1")
        clean["prediction_meta_boundary"] = "process_anchor_metadata_only_no_teacher_answer_or_ocr_text"
        clean["teacher_label_is_scaffold"] = False
        clean["used_in_strict_teacher_off_input"] = False
        clean["answer_table_lookup"] = False
        clean["full_string_or_sentence_action"] = False
        return clean
