from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


class TextActionActuator:
    """
    Transitional APV2.1 text actuator.

    It does not replace the Bn/Cn definitions. It sits after cognition and
    turns selected action tendencies plus predicted text labels into an explicit
    write / reread / revise trace so the system can start accumulating real
    output-side evidence.
    """

    def __init__(self, *, max_visible_buffer: int = 12) -> None:
        self.max_visible_buffer = max(4, int(max_visible_buffer))
        self._visible_tokens: list[dict] = []
        self._revision_events: list[dict] = []
        self._recent_events: list[dict] = []
        self._cursor_index: int = 0

    def step(
        self,
        *,
        tick_index: int,
        input_text: str,
        selected_actions: list[dict],
        fast_cn: list[dict],
        slow_cn: list[dict],
        focus_labels: list[str],
        cognitive_feelings: dict,
    ) -> dict:
        input_token = str(input_text or "").strip()
        previous_token = self._visible_tokens[-1]["token"] if self._visible_tokens else ""
        expected_token_info = self._pick_expected_token_info(
            fast_cn=fast_cn,
            slow_cn=slow_cn,
            exclude_token=input_token,
            visible_text=self._visible_text(),
        )
        expected_token = str(expected_token_info.get("token", "") or "")
        action_ids = [str(item.get("action_id", "") or "") for item in (selected_actions or [])]
        dissonance = float((cognitive_feelings.get("channels", {}) or {}).get("dissonance", 0.0) or 0.0)
        surprise = float((cognitive_feelings.get("channels", {}) or {}).get("surprise", 0.0) or 0.0)
        pressure = float((cognitive_feelings.get("channels", {}) or {}).get("pressure", 0.0) or 0.0)
        events = []
        output_items = []
        revision_detected = False
        primary_action = self._primary_text_action(selected_actions)

        if primary_action and not input_token:
            action_id = str(primary_action.get("action_id", "") or "")
            params = dict(primary_action.get("params", {}) or {})
            before_text = self._visible_text()
            cursor_before = self._bounded_cursor(params.get("cursor", self._cursor_index))
            if action_id == "action::text_insert":
                token = str(params.get("token", params.get("text", "")) or "") or expected_token
                if token:
                    cursor = self._bounded_cursor(params.get("cursor", self._cursor_index))
                    action_param_reason = str(params.get("reason", "") or "")
                    action_param_kind = str(params.get("parameter_kind", "") or "")
                    candidate_token = str(params.get("candidate_token", token) or token)
                    action_expected_token = str(params.get("expected_token", "") or "")
                    event = {
                        "tick_index": int(tick_index),
                        "event_type": "insert",
                        "token": token,
                        "expected_token": expected_token,
                        "candidate_token": candidate_token,
                        "source": action_id,
                        "notes": ["direct_text_insert_action"],
                        "action_id": action_id,
                        "cursor_before": cursor,
                        "cursor_after": cursor + 1,
                        "visible_text_before": before_text,
                    }
                    if action_param_reason:
                        event["action_param_reason"] = action_param_reason
                    if action_param_kind:
                        event["action_param_kind"] = action_param_kind
                    if action_expected_token:
                        event["action_expected_token"] = action_expected_token
                    self._visible_tokens.insert(cursor, event)
                    self._cursor_index = min(len(self._visible_tokens), cursor + 1)
                    # Keep the real draft surface intact. `max_visible_buffer`
                    # only limits short-term self-observation/recent-event
                    # windows; truncating the draft itself breaks long
                    # charwise successor traces and commits only the tail.
                    self._cursor_index = self._bounded_cursor(self._cursor_index)
                    event["visible_text_after"] = self._visible_text()
                    events.append(event)
                    output_items.append(self._text_item(label=f"text_action::insert::{token}", display=f"insert:{token}", energy=0.44, event=event))
            elif action_id == "action::text_delete":
                span = self._resolve_span(params.get("span"), default_to_cursor_previous=True)
                deleted_tokens = self._visible_tokens[span[0] : span[1]]
                deleted = "".join(str(row.get("token", "") or "") for row in deleted_tokens)
                if span[1] > span[0]:
                    del self._visible_tokens[span[0] : span[1]]
                self._cursor_index = self._bounded_cursor(span[0])
                event = {
                    "tick_index": int(tick_index),
                    "event_type": "delete",
                    "token": deleted,
                    "source": action_id,
                    "notes": ["direct_text_delete_action"],
                    "action_id": action_id,
                    "span": list(span),
                    "cursor_before": cursor_before,
                    "cursor_after": self._cursor_index,
                    "visible_text_before": before_text,
                    "visible_text_after": self._visible_text(),
                }
                events.append(event)
                output_items.append(self._text_item(label=f"text_action::delete::{deleted or 'empty'}", display=f"delete:{deleted}", energy=0.38, event=event))
            elif action_id == "action::text_replace":
                explicit_span = params.get("span")
                span = self._resolve_span(explicit_span, default_to_cursor_previous=True)
                target_index, mismatch_row = self._latest_mismatch_token()
                if explicit_span is None:
                    if target_index < 0 and self._visible_tokens:
                        target_index = max(0, min(len(self._visible_tokens) - 1, self._cursor_index - 1))
                        mismatch_row = self._visible_tokens[target_index]
                    span = (target_index, target_index + 1) if target_index >= 0 else (self._cursor_index, self._cursor_index)
                replacement = str(params.get("new_text", params.get("token", "")) or "") or expected_token
                old = "".join(str(row.get("token", "") or "") for row in self._visible_tokens[span[0] : span[1]])
                if explicit_span is None and target_index < 0 and not old:
                    event = {
                        "tick_index": int(tick_index),
                        "event_type": "replace_noop",
                        "from_token": "",
                        "to_token": replacement,
                        "expected_token": str(params.get("expected_token", replacement) or replacement),
                        "candidate_token": str(params.get("candidate_token", replacement) or replacement),
                        "source": action_id,
                        "notes": [
                            "direct_text_replace_action_noop",
                            "replace_requires_existing_target_or_explicit_span",
                            "empty_draft_replace_does_not_become_insert",
                        ],
                        "action_id": action_id,
                        "target_index": -1,
                        "conflict_index": -1,
                        "span": list(span),
                        "cursor_before": cursor_before,
                        "cursor_after": self._cursor_index,
                        "visible_text_before": before_text,
                        "visible_text_after": before_text,
                    }
                    events.append(event)
                    output_items.append(
                        self._text_item(
                            label="text_action::replace_noop",
                            display="replace_noop",
                            energy=0.16,
                            event=event,
                            virtual_energy=0.12,
                        )
                    )
                    self._remember_events(events)
                    return {
                        "visible_tokens": list(self._visible_tokens),
                        "visible_text": self._visible_text(),
                        "cursor_index": int(self._cursor_index),
                        "recent_events": events,
                        "revision_events": list(self._revision_events),
                        "output_items": output_items,
                        "revision_detected": False,
                        "focus_labels": list(focus_labels or []),
                        "expected_token": expected_token,
                    }
                replacement_units = self._split_replacement_text(replacement) if explicit_span is not None else ([replacement] if replacement else [])
                replacement_entries = [
                    {
                        "tick_index": int(tick_index),
                        "event_type": "write_revision",
                        "token": token,
                        "expected_token": replacement,
                        "source": "text_actuator_direct_replace",
                        "notes": ["direct_replace_commit"],
                    }
                    for token in replacement_units
                ]
                if span[1] > span[0] and replacement_entries:
                    revision_detected = True
                    self._visible_tokens[span[0] : span[1]] = replacement_entries
                elif replacement:
                    self._visible_tokens[span[0] : span[1]] = replacement_entries
                    revision_detected = True
                self._cursor_index = self._bounded_cursor(span[0] + len(replacement_entries))
                event = {
                    "tick_index": int(tick_index),
                    "event_type": "replace",
                    "from_token": old,
                    "to_token": replacement,
                    "expected_token": str(params.get("expected_token", replacement) or replacement),
                    "candidate_token": str(params.get("candidate_token", replacement) or replacement),
                    "source": action_id,
                    "notes": ["direct_text_replace_action"],
                    "action_id": action_id,
                    "target_index": int(span[0]),
                    "conflict_index": int(span[0]),
                    "span": list(span),
                    "cursor_before": cursor_before,
                    "cursor_after": self._cursor_index,
                    "visible_text_before": before_text,
                    "visible_text_after": self._visible_text(),
                }
                self._revision_events.append(event)
                self._revision_events = self._revision_events[-self.max_visible_buffer :]
                events.append(event)
                output_items.append(
                    self._text_item(
                        label=f"text_action::replace::{replacement or 'empty'}",
                        display=f"replace:{old}->{replacement}",
                        energy=0.5,
                        event=event,
                        virtual_energy=0.18,
                    )
                )
            elif action_id == "action::text_commit":
                visible_text = "".join(entry["token"] for entry in self._visible_tokens if str(entry.get("token", "") or ""))
                clear_after_commit = bool(params.get("clear_after_commit", True))
                event = {
                    "tick_index": int(tick_index),
                    "event_type": "commit",
                    "token": visible_text,
                    "target_channel": str(params.get("target_channel", "text_buffer") or "text_buffer"),
                    "source": action_id,
                    "action_id": action_id,
                    "cursor_before": cursor_before,
                    "visible_text_before": before_text,
                    "clear_after_commit": clear_after_commit,
                    "notes": [
                        "direct_text_commit_action",
                        "commit_is_internal_buffer_trace_only",
                        "commit_clears_visible_text_buffer" if clear_after_commit else "commit_keeps_visible_text_buffer",
                    ],
                }
                if clear_after_commit:
                    self._visible_tokens = []
                    self._cursor_index = 0
                event["cursor_after"] = self._cursor_index
                event["visible_text_after"] = self._visible_text()
                events.append(event)
                output_items.append(self._text_item(label="text_action::commit", display="commit", energy=0.52, event=event))
                if visible_text:
                    output_items.append(
                        self._text_item(
                            label=f"text_action::sent::{visible_text}",
                            display=f"sent:{visible_text}",
                            energy=0.32,
                            event={**event, "event_type": "sent_memory", "sent_text": visible_text},
                        )
                    )
            elif action_id == "action::text_reread":
                span = self._resolve_span(params.get("span"), default_to_cursor_previous=False)
                reread_text = self._span_text(span)
                # Rereading is self-observation of the draft surface. It must
                # still work when no next token is currently predicted; people
                # can look back at what they wrote before knowing what to add.
                reread_token = reread_text
                if reread_token:
                    event = {
                        "tick_index": int(tick_index),
                        "event_type": "reread",
                        "token": reread_token,
                        "source": action_id,
                        "notes": ["direct_text_reread_action"],
                        "action_id": action_id,
                        "span": list(span),
                        "cursor_before": cursor_before,
                        "cursor_after": self._cursor_index,
                        "visible_text_before": before_text,
                        "visible_text_after": before_text,
                    }
                    events.append(event)
                    output_items.append(self._text_item(label=f"text_action::reread::{reread_token}", display=f"reread:{reread_token}", energy=0.24, event=event))
                    output_items.extend(self._draft_read_items(span=span, event=event))
        elif input_token:
            kind = "external_read"
            notes = ["external_text_read_into_input_channel", "not_ap_visible_draft"]
            event = {
                "tick_index": int(tick_index),
                "event_type": kind,
                "token": input_token,
                "expected_token": expected_token,
                "source": "external_text",
                "notes": notes,
            }
            events.append(event)
            output_items.append(
                {
                    "sa_label": f"text_input::external_read::{input_token}",
                    "display_text": f"读入:{input_token}",
                    "family": "text_input",
                    "source_type": "external_text_readback",
                    "real_energy": 0.24,
                    "virtual_energy": 0.04,
                    "anchor_meta": dict(event),
                }
            )
        elif "action::replay_recent_context" in action_ids and expected_token:
            mismatch_index, mismatch_row = self._latest_mismatch_token()
            # Prefer revising the latest mismatch we actually observed (wrong-then-correct),
            # rather than only comparing to the newest visible token.
            revision_target = str((mismatch_row or {}).get("expected_token", "") or expected_token)
            mismatch_token = str((mismatch_row or {}).get("token", "") or "")
            should_revise = bool(mismatch_row) and mismatch_token and revision_target and mismatch_token != revision_target
            if should_revise and (dissonance >= 0.6 or surprise >= 0.6):
                revision_detected = True
                event = {
                    "tick_index": int(tick_index),
                    "event_type": "revise",
                    "from_token": mismatch_token,
                    "to_token": revision_target,
                    "source": "action::replay_recent_context",
                    "notes": ["reread_then_revise", "predicted_token_restore"],
                    "target_index": int(mismatch_index),
                }
                # Overwrite the wrong token in-place (true revision, not append).
                self._visible_tokens[mismatch_index] = {
                    **dict(self._visible_tokens[mismatch_index] or {}),
                    "event_type": "write_revision",
                    "token": revision_target,
                    "expected_token": revision_target,
                    "source": "text_actuator_revision",
                    "notes": list((self._visible_tokens[mismatch_index].get("notes", []) or [])) + ["revision_commit"],
                }
                self._cursor_index = self._bounded_cursor(mismatch_index + 1)
                self._revision_events.append(event)
                self._revision_events = self._revision_events[-self.max_visible_buffer :]
                events.append(event)
                output_items.append(
                    {
                        "sa_label": f"text_action::revise::{revision_target}",
                        "display_text": f"改写:{mismatch_token}->{revision_target}",
                        "family": "text_action",
                        "source_type": "text_action",
                        "real_energy": 0.48,
                        "virtual_energy": 0.24,
                        "anchor_meta": dict(event),
                    }
                )
            else:
                event = {
                    "tick_index": int(tick_index),
                    "event_type": "reread",
                    "token": expected_token,
                    "source": "action::replay_recent_context",
                    "notes": ["context_reread"],
                }
                events.append(event)
                output_items.append(
                    {
                        "sa_label": f"text_action::reread::{expected_token}",
                        "display_text": f"回读:{expected_token}",
                        "family": "text_action",
                        "source_type": "text_action",
                        "real_energy": 0.22,
                        "anchor_meta": dict(event),
                    }
                )
                output_items.extend(self._draft_read_items(span=(0, len(self._visible_tokens)), event=event))
        elif (
            "action::continue_focus" in action_ids
            and expected_token
            and self._visible_tokens
            and str(expected_token_info.get("alignment", "") or "") == "aligned"
        ):
            event = {
                "tick_index": int(tick_index),
                "event_type": "prepare_continue",
                "token": expected_token,
                "source": "action::continue_focus",
                "notes": ["focus_continuation_prepare", "aligned_visible_draft_successor"],
            }
            events.append(event)
            output_items.append(
                {
                    "sa_label": f"text_action::prepare::{expected_token}",
                    "display_text": f"续写准备:{expected_token}",
                    "family": "text_action",
                    "source_type": "text_action",
                    "virtual_energy": _round4(0.18 + pressure * 0.08),
                    "anchor_meta": dict(event),
                }
            )

        visible_text = self._visible_text()
        self._remember_events(events)
        draft_state_event = self._draft_state_event()
        if draft_state_event:
            output_items.append(
                self._text_item(
                    label="text_action::draft_state",
                    display=f"draft:{visible_text}",
                    energy=0.16,
                    event=draft_state_event,
                    virtual_energy=0.04,
                )
            )
        return {
            "visible_tokens": list(self._visible_tokens),
            "visible_text": visible_text,
            "cursor_index": int(self._cursor_index),
            "recent_events": events,
            "revision_events": list(self._revision_events),
            "output_items": output_items,
            "revision_detected": revision_detected,
            "focus_labels": list(focus_labels or []),
            "expected_token": expected_token,
        }

    def visible_text(self) -> str:
        return self._visible_text()

    def mark_external_turn_boundary(self, *, tick_index: int, reason: str = "new_external_text_turn") -> dict:
        """
        Separate a newly read user turn from the previous committed draft trace.

        The committed text has already entered the normal memory stream. Keeping
        the actuator's local recent insert/reread/commit events as "current
        draft" evidence after the visible surface is empty makes the next turn
        look like a stale continuation of the old reply. If a draft is still
        visible, preserve it so the user can interrupt and AP can reread/revise.
        """

        if self._visible_tokens:
            return {
                "schema_id": "text_action_external_turn_boundary/v1",
                "applied": False,
                "reason": "visible_draft_preserved",
                "boundary_reason": str(reason or ""),
                "visible_length": len(self._visible_tokens),
                "recent_event_count": len(self._recent_events),
            }
        recent_events = [dict(row) for row in self._recent_events if isinstance(row, dict)]
        has_committed_surface = any(str(row.get("event_type", "") or "") == "commit" for row in recent_events)
        if not has_committed_surface:
            return {
                "schema_id": "text_action_external_turn_boundary/v1",
                "applied": False,
                "reason": "no_committed_empty_surface_residue",
                "boundary_reason": str(reason or ""),
                "visible_length": 0,
                "recent_event_count": len(recent_events),
            }
        cleared_event_count = len(self._recent_events)
        cleared_revision_count = len(self._revision_events)
        self._recent_events = []
        self._revision_events = []
        self._cursor_index = 0
        return {
            "schema_id": "text_action_external_turn_boundary/v1",
            "applied": True,
            "reason": "committed_empty_surface_residue_cleared_for_new_turn",
            "boundary_reason": str(reason or ""),
            "tick_index": int(tick_index),
            "cleared_recent_event_count": int(cleared_event_count),
            "cleared_revision_event_count": int(cleared_revision_count),
        }

    def short_term_context_items(self) -> list[dict]:
        """
        Return the actuator's recent draft events as lightweight state items.

        The visible text buffer is AP's own draft surface, so planner needs a
        short-term memory of recent write/reread/revision events even when
        those items are no longer salient enough to survive the state snapshot
        limit. This mirrors a person remembering what they just typed while
        deciding whether to reread or revise it.
        """

        items = []
        visible_window = self._visible_tokens[-self.max_visible_buffer :]
        for visible_offset, row in enumerate(visible_window):
            if not isinstance(row, dict):
                continue
            token = str(row.get("token", "") or "")
            event_type = str(row.get("event_type", "") or "")
            if not token or not event_type:
                continue
            items.append(
                self._text_item(
                    label=f"text_action::write::{token}",
                    display=f"write:{token}",
                    energy=0.18 if self._is_mismatch_row(row) else 0.10,
                    event=row,
                    virtual_energy=0.0,
                )
            )
            if event_type == "insert":
                # Keep the actuator-side event vocabulary aligned with the
                # state item written immediately after a direct text_insert.
                # This lets later ticks learn from "I inserted this token" as
                # well as the more generic "this token is on my draft surface".
                items.append(
                    self._text_item(
                        label=f"text_action::insert::{token}",
                        display=f"insert:{token}",
                        energy=0.12,
                        event=row,
                        virtual_energy=0.0,
                    )
                )
            visible_index = max(0, visible_offset)
            try:
                visible_index = max(0, int(row.get("cursor_after", 0) or 0) - 1)
            except (TypeError, ValueError):
                visible_index = 0
            visible_anchor = {
                "schema_id": "text_visible_draft_token/v1",
                "event_type": "visible_draft_token",
                "token": token,
                "position": int(visible_index),
                "current_glyph_index": int(visible_index),
                "cursor_index": int(self._cursor_index),
                "visible_length": len(self._visible_tokens),
                "visible_text": self._visible_text(),
                "last_visible_token": str((self._visible_tokens[-1] if self._visible_tokens else {}).get("token", "") or ""),
                "source_event_type": event_type,
                "self_generated": str(row.get("source", "") or "") != "external_text",
                "current_read_tick": True,
                "process_anchor_role": "internal_draft_visible_prefix",
                "prediction_payload_priority": "previous_prefix_context",
                "previous_prefix": self._visible_text(),
                "meaning": "AP self-observes a visible token on its own draft surface for successor continuation",
            }
            items.append(
                {
                    "sa_label": f"text::{token}",
                    "display_text": token,
                    "family": "text",
                    "source_type": "internal_draft_visible",
                    "real_energy": 0.14,
                    "virtual_energy": 0.03,
                    "cognitive_pressure": 0.03,
                    "anchor_meta": visible_anchor,
                }
            )
        visible_text = self._visible_text()
        if visible_text:
            draft_state_event = self._draft_state_event()
            if draft_state_event:
                items.append(
                    self._text_item(
                        label="text_action::draft_state",
                        display=f"draft:{visible_text}",
                        energy=0.34,
                        event=draft_state_event,
                        virtual_energy=0.10,
                    )
                )
            last_token = str((self._visible_tokens[-1] if self._visible_tokens else {}).get("token", "") or "")
            cursor_event = {
                "schema_id": "text_cursor_state/v1",
                "event_type": "cursor_state",
                "cursor_index": int(self._cursor_index),
                "visible_length": len(self._visible_tokens),
                "last_visible_token": last_token,
                "has_visible_text": True,
                "meaning": "low-grain draft position context for continuing or repairing text output",
            }
            items.append(
                self._text_item(
                    label=f"text_action::cursor_index::{self._cursor_index}",
                    display=f"cursor:{self._cursor_index}",
                    energy=0.09,
                    event=cursor_event,
                    virtual_energy=0.0,
                )
            )
            continuation_event = {
                "schema_id": "text_revision_opportunity/v1",
                "event_type": "next_unread_region_pressure",
                "operation": "insert",
                "conflict_kind": "continue_after_visible_prefix",
                "cursor": int(self._cursor_index),
                "span": [int(self._cursor_index), int(self._cursor_index)],
                "visible_text": visible_text,
                "visible_length": len(self._visible_tokens),
                "last_visible_token": last_token,
                "support": 0.22,
                "notes": [
                    "self_observed_draft_continuation_pressure",
                    "candidate_text_absent_until_memory_prediction",
                    "not_answer_hint",
                ],
            }
            items.append(
                {
                    "sa_label": "text_revision_opportunity::continue_after_visible_prefix",
                    "display_text": "continue_after_visible_prefix",
                    "family": "text_revision_opportunity",
                    "source_type": "text_action",
                    "real_energy": 0.10,
                    "virtual_energy": 0.16,
                    "cognitive_pressure": 0.08,
                    "anchor_meta": continuation_event,
                }
            )
        else:
            empty_state = self.draft_state()
            items.append(
                self._text_item(
                    label="text_action::draft_state",
                    display="draft:",
                    energy=0.12,
                    event={
                        **empty_state,
                        "process_anchor_role": "empty_draft_start_readout_context",
                        "meaning": "AP self-observes an empty draft surface; this can cooccur with looking for the first unresolved readout slot without providing any target character.",
                    },
                    virtual_energy=0.02,
                )
            )
            items.append(
                {
                    "sa_label": "text_revision_opportunity::start_empty_draft",
                    "display_text": "start_empty_draft",
                    "family": "text_revision_opportunity",
                    "source_type": "text_action",
                    "real_energy": 0.10,
                    "virtual_energy": 0.14,
                    "cognitive_pressure": 0.08,
                    "anchor_meta": {
                        "schema_id": "text_revision_opportunity/v1",
                        "event_type": "start_empty_draft",
                        "operation": "insert",
                        "conflict_kind": "start_empty_draft",
                        "cursor": 0,
                        "span": [0, 0],
                        "visible_text": "",
                        "visible_length": 0,
                        "support": 0.20,
                        "process_anchor_role": "empty_draft_first_unread_slot_pressure",
                        "used_in_strict_teacher_off_input": False,
                        "answer_table_lookup": False,
                        "full_string_or_sentence_action": False,
                        "notes": [
                            "self_observed_empty_draft_start_pressure",
                            "candidate_text_absent_until_memory_prediction",
                            "not_answer_hint",
                        ],
                    },
                }
            )
        for row in self._revision_events[-self.max_visible_buffer :]:
            if not isinstance(row, dict):
                continue
            token = str(row.get("to_token", row.get("token", "")) or "")
            event = dict(row)
            event.setdefault("event_type", "revise")
            items.append(
                self._text_item(
                    label=f"text_action::revise::{token or 'empty'}",
                    display=f"revise:{token}",
                    energy=0.16,
                    event=event,
                    virtual_energy=0.0,
                )
            )
        for row in self._recent_events[-self.max_visible_buffer :]:
            if not isinstance(row, dict):
                continue
            event_type = str(row.get("event_type", "") or "")
            if event_type not in {"reread", "commit"}:
                continue
            token = str(row.get("token", "") or "")
            label_token = token or event_type
            items.append(
                self._text_item(
                    label=f"text_action::{event_type}::{label_token}",
                    display=f"{event_type}:{token}",
                    energy=0.12 if event_type == "reread" else 0.14,
                    event=row,
                    virtual_energy=0.0,
                )
            )
            if event_type == "commit" and token:
                event = dict(row)
                event["event_type"] = "sent_memory"
                event["sent_text"] = token
                items.append(
                    self._text_item(
                        label=f"text_action::sent::{token}",
                        display=f"sent:{token}",
                        energy=0.11,
                        event=event,
                        virtual_energy=0.0,
                    )
                )
        draft_state = self._draft_state_event()
        if draft_state:
            # This item is a compact planning handle, not a concept sample. It
            # lets the drive manager know whether a draft is underway, whether
            # it was just reread, and whether commit readiness is plausible.
            items.append(
                self._text_item(
                    label="text_action::draft_state",
                    display="draft_state",
                    energy=0.08,
                    event=draft_state,
                    virtual_energy=0.0,
                )
            )
        return items

    def parameter_events(self, events: list[dict]) -> list[dict]:
        rows = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            action_id = str(event.get("action_id", event.get("source", "")) or "")
            event_type = str(event.get("event_type", "") or "")
            if action_id == "action::text_insert" or event_type == "insert":
                rows.append(dict(event, action_id="action::text_insert", parameter_kind="text_insert"))
            elif action_id == "action::text_delete" or event_type == "delete":
                rows.append(dict(event, action_id="action::text_delete", parameter_kind="text_delete"))
            elif action_id == "action::text_replace" or event_type == "replace":
                rows.append(dict(event, action_id="action::text_replace", parameter_kind="text_replace"))
        return rows

    def apply_feedback_to_recent_action(self, feedback: dict, *, causal_window: dict | None = None) -> dict:
        observed = dict(feedback or {})
        if not observed:
            return {"schema_id": "text_action_feedback_binding/v1", "applied": False, "reason": "empty_feedback"}
        reward = max(0.0, float(observed.get("reward", 0.0) or 0.0))
        punishment = max(0.0, float(observed.get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float(observed.get("correctness", 0.0) or 0.0))
        if reward <= 0.0 and punishment <= 0.0 and correctness <= 0.0:
            return {"schema_id": "text_action_feedback_binding/v1", "applied": False, "reason": "no_reward_punishment_or_correctness"}
        window = dict(causal_window or {})
        target_events = [
            dict(row)
            for row in list(window.get("text_parameter_events", []) or [])
            if isinstance(row, dict)
            and str(row.get("parameter_kind", "") or "") in {"text_insert", "text_replace"}
            and str(row.get("token", "") or row.get("to_token", "") or "")
        ]
        if not target_events:
            return {"schema_id": "text_action_feedback_binding/v1", "applied": False, "reason": "no_text_parameter_event"}
        target = target_events[-1]
        token = str(target.get("token", "") or target.get("to_token", "") or "")
        tick_index = target.get("tick_index")
        expected_token = str(target.get("expected_token", "") or "")
        action_expected_token = str(target.get("action_expected_token", "") or target.get("candidate_token", "") or "")
        explicit_feedback_target = str(
            observed.get("feedback_expected_token", "")
            or observed.get("expected_token", "")
            or observed.get("teacher_reference_token_post_action_only", "")
            or observed.get("target_token", "")
            or ""
        )
        action_feedback_target = str(explicit_feedback_target or action_expected_token or expected_token or token or "")
        feedback_expected_token = str(
            explicit_feedback_target
            or action_expected_token
            or ""
        )
        mismatch_basis = "explicit_post_action_feedback_target" if explicit_feedback_target else (
            "action_expected_token_feedback_target" if action_expected_token else ""
        )
        token_mismatch = bool(token and feedback_expected_token and token != feedback_expected_token)
        outcome = "punished" if token_mismatch or punishment > max(reward, correctness) else "rewarded"
        marker = {
            "feedback_outcome": outcome,
            "feedback_reward": _round4(reward),
            "feedback_punishment": _round4(punishment),
            "feedback_correctness": _round4(correctness),
            "feedback_token_mismatch": bool(token_mismatch),
            "feedback_expected_token": action_feedback_target,
            "feedback_reference_token": feedback_expected_token,
            "feedback_mismatch_basis": mismatch_basis,
            "observed_feedback": observed,
            "feedback_binding_schema": "text_action_feedback_binding/v1",
            "feedback_binding_notes": [
                "post_action_feedback_bound_to_recent_text_action",
                "token_mismatch_requires_explicit_post_action_reference",
                "punished_text_is_repair_evidence_not_positive_prediction" if outcome == "punished" else "rewarded_text_can_remain_positive_prediction",
            ],
        }
        updated_count = 0
        for row in list(self._visible_tokens) + list(self._recent_events):
            if not isinstance(row, dict):
                continue
            row_token = str(row.get("token", "") or row.get("to_token", "") or "")
            if row_token != token:
                continue
            if tick_index is not None and str(row.get("tick_index", "") or "") != str(tick_index):
                continue
            row.update(marker)
            notes = list(row.get("notes", []) or [])
            for note in marker["feedback_binding_notes"]:
                if note not in notes:
                    notes.append(note)
            if outcome == "punished" and "feedback_punished_text_candidate" not in notes:
                notes.append("feedback_punished_text_candidate")
            row["notes"] = notes
            updated_count += 1
        return {
            "schema_id": "text_action_feedback_binding/v1",
            "applied": bool(updated_count > 0),
            "updated_count": int(updated_count),
            "target_token": token,
            "target_tick_index": tick_index,
            "expected_token": expected_token,
            "action_expected_token": action_expected_token,
            "action_feedback_target": action_feedback_target,
            "feedback_reference_token": feedback_expected_token,
            "token_mismatch": bool(token_mismatch),
            "mismatch_basis": mismatch_basis,
            "feedback_outcome": outcome,
            "observed_feedback": observed,
            "boundary": "feedback_marks_text_action_consequence_without_deleting_memory_or_using_teacher_answer_in_probe",
        }

    def _primary_text_action(self, selected_actions: list[dict]) -> dict:
        direct_ids = {"action::text_insert", "action::text_delete", "action::text_replace", "action::text_commit", "action::text_reread"}
        for row in selected_actions or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("action_id", "") or "") in direct_ids:
                return dict(row)
        return {}

    def _text_item(self, *, label: str, display: str, energy: float, event: dict, virtual_energy: float = 0.0) -> dict:
        item = {
            "sa_label": str(label),
            "display_text": str(display),
            "family": "text_action",
            "source_type": "text_action",
            "real_energy": _round4(energy),
            "anchor_meta": dict(event),
        }
        if virtual_energy > 0.0:
            item["virtual_energy"] = _round4(virtual_energy)
        return item

    def _is_mismatch_row(self, row: dict) -> bool:
        event_type = str(row.get("event_type", "") or "")
        if event_type == "write_mismatch":
            return True
        if bool(row.get("feedback_token_mismatch", False)):
            return True
        outcome = str(row.get("feedback_outcome", "") or "")
        reference = str(
            row.get("feedback_reference_token", "")
            or row.get("teacher_reference_token_post_action_only", "")
            or row.get("target_token", "")
            or ""
        )
        return bool(outcome == "punished" and reference)

    def _remember_events(self, events: list[dict]) -> None:
        for event in events or []:
            if isinstance(event, dict):
                self._recent_events.append(dict(event))
        self._recent_events = self._recent_events[-max(self.max_visible_buffer * 2, 8) :]

    def draft_state(self, *, current_tick: int | None = None) -> dict:
        """
        Return AP's current visible draft surface as readback material.

        This is a read-only self-observation API for planners and feelings. It
        does not expose a teacher answer or mutate the draft buffer.
        """

        event = self._draft_state_event()
        if not event:
            empty = {
                "schema_id": "text_draft_state/v1",
                "event_type": "draft_state",
                "active_draft_surface": True,
                "visible_text": "",
                "visible_tokens": [],
                "visible_length": 0,
                "cursor_index": 0,
                "last_visible_token": "",
                "insert_count": 0,
                "reread_count": 0,
                "revision_count": 0,
                "last_event_tick": -1,
                "last_insert_tick": -1,
                "last_reread_tick": -1,
                "last_commit_tick": -1,
                "notes": ["empty_draft_state_planning_context"],
            }
            if current_tick is not None:
                for key in (
                    "last_event_tick",
                    "last_insert_tick",
                    "last_reread_tick",
                    "last_delete_tick",
                    "last_replace_tick",
                    "last_revision_tick",
                    "last_mutation_tick",
                    "last_commit_tick",
                ):
                    empty.setdefault(key, -1)
                    empty[key.replace("_tick", "_age")] = 9999
            return empty
        if current_tick is None:
            return dict(event)
        stamped = dict(event)
        for key in (
            "last_event_tick",
            "last_insert_tick",
            "last_reread_tick",
            "last_delete_tick",
            "last_replace_tick",
            "last_revision_tick",
            "last_mutation_tick",
            "last_commit_tick",
        ):
            try:
                tick = int(stamped.get(key, -1) or -1)
            except (TypeError, ValueError):
                tick = -1
            stamped[key] = tick
            stamped[key.replace("_tick", "_age")] = 9999 if tick < 0 else max(0, int(current_tick) - tick)
        return stamped

    def _draft_state_event(self) -> dict:
        visible_text = self._visible_text()
        if not visible_text and not self._recent_events and not self._revision_events:
            return {}
        events = [dict(row) for row in self._recent_events if isinstance(row, dict)]
        visible_rows = [dict(row) for row in self._visible_tokens if isinstance(row, dict)]

        def _event_type(row: dict) -> str:
            return str(row.get("event_type", "") or "")

        def _event_tick(row: dict) -> int:
            try:
                return int(row.get("tick_index", -1) or -1)
            except (TypeError, ValueError):
                return -1

        def _last_tick(types: set[str]) -> int:
            return max([_event_tick(row) for row in events if _event_type(row) in types] or [-1])

        insert_count = sum(1 for row in events if _event_type(row) == "insert")
        external_write_count = sum(1 for row in events if _event_type(row) in {"write", "write_mismatch"} and str(row.get("source", "") or "") == "external_text")
        # AP's internal expected_token can disagree with a teacher-on or
        # scaffolded token without proving the visible draft is wrong. Hard
        # mismatch evidence requires explicit write_mismatch or post-action
        # feedback, matching the V2 process-anchor boundary.
        mismatch_count = sum(1 for row in visible_rows if self._is_mismatch_row(row))
        revision_count = len(self._revision_events) + sum(1 for row in visible_rows if _event_type(row) == "write_revision")
        reread_count = sum(1 for row in events if _event_type(row) == "reread")
        delete_count = sum(1 for row in events if _event_type(row) == "delete")
        replace_count = sum(1 for row in events if _event_type(row) == "replace")
        commit_count = sum(1 for row in events if _event_type(row) == "commit")
        last_event = dict(events[-1]) if events else {}
        visible_tokens = [str(row.get("token", "") or "") for row in visible_rows if str(row.get("token", "") or "")]
        trailing_repeat_token = ""
        trailing_repeat_count = 0
        if visible_tokens:
            trailing_repeat_token = visible_tokens[-1]
            for token in reversed(visible_tokens):
                if token != trailing_repeat_token:
                    break
                trailing_repeat_count += 1
        duplicate_ratio = 0.0
        if visible_tokens:
            duplicate_ratio = 1.0 - (len(set(visible_tokens)) / max(1, len(visible_tokens)))
        mutation_types = {"insert", "delete", "replace", "revise", "write_revision", "commit"}
        revision_events = [dict(row) for row in self._revision_events if isinstance(row, dict)]
        latest_mismatch_index = -1
        latest_mismatch_token = ""
        latest_mismatch_expected_token = ""
        latest_mismatch_tick = -1
        for index in range(len(visible_rows) - 1, -1, -1):
            row = visible_rows[index]
            if self._is_mismatch_row(row):
                token = str(row.get("token", "") or "")
                expected = str(
                    row.get("feedback_reference_token", "")
                    or row.get("expected_token", "")
                    or ""
                )
                latest_mismatch_index = index
                latest_mismatch_token = token
                latest_mismatch_expected_token = expected
                latest_mismatch_tick = _event_tick(row)
                break
        return {
            "schema_id": "text_draft_state/v1",
            "event_type": "draft_state",
            "active_draft_surface": True,
            "visible_text": visible_text,
            "visible_tokens": visible_tokens,
            "visible_length": len(self._visible_tokens),
            "cursor_index": int(self._cursor_index),
            "last_visible_token": str((self._visible_tokens[-1] if self._visible_tokens else {}).get("token", "") or ""),
            "trailing_repeat_token": trailing_repeat_token,
            "trailing_repeat_count": int(trailing_repeat_count),
            "duplicate_ratio": _round4(duplicate_ratio),
            "insert_count": int(insert_count),
            "external_write_count": int(external_write_count),
            "mismatch_count": int(mismatch_count),
            "revision_count": int(revision_count),
            "reread_count": int(reread_count),
            "delete_count": int(delete_count),
            "replace_count": int(replace_count),
            "commit_count": int(commit_count),
            "last_event_type": _event_type(last_event),
            "last_event_tick": _event_tick(last_event),
            "last_insert_tick": _last_tick({"insert"}),
            "last_reread_tick": _last_tick({"reread"}),
            "last_delete_tick": _last_tick({"delete"}),
            "last_replace_tick": _last_tick({"replace"}),
            "last_revision_tick": max([_event_tick(row) for row in revision_events] or [-1]),
            "last_mutation_tick": _last_tick(mutation_types),
            "last_commit_tick": _last_tick({"commit"}),
            "latest_mismatch_index": int(latest_mismatch_index),
            "latest_mismatch_tick": int(latest_mismatch_tick),
            "latest_mismatch_token": latest_mismatch_token,
            "latest_mismatch_expected_token": latest_mismatch_expected_token,
            "notes": ["draft_state_planning_context"],
        }

    def _draft_read_items(self, *, span: tuple[int, int], event: dict) -> list[dict]:
        items = []
        start, end = span
        for offset, row in enumerate(self._visible_tokens[start:end]):
            token = str((row or {}).get("token", "") or "")
            if not token:
                continue
            anchor = dict(event)
            anchor.update(
                {
                    "event_type": "draft_read_token",
                    "token": token,
                    "position": int(start + offset),
                    "source_event_type": str(event.get("event_type", "") or ""),
                    "self_generated": str((row or {}).get("source", "") or "") != "external_text",
                }
            )
            for key in ("feedback_outcome", "feedback_reward", "feedback_punishment", "feedback_correctness", "feedback_binding_schema"):
                if key in row:
                    anchor[key] = row.get(key)
            if str(anchor.get("feedback_outcome", "") or "") == "punished":
                notes = list(anchor.get("notes", []) or [])
                if "draft_read_token_was_punished_text_candidate" not in notes:
                    notes.append("draft_read_token_was_punished_text_candidate")
                anchor["notes"] = notes
                anchor["prediction_payload_role"] = "negative_feedback_repair_context"
            # Reread text is how AP sees its own draft again. It re-enters the
            # ordinary text channel at low energy so successor recall can
            # continue from self-written words without treating action logs as
            # concepts.
            items.append(
                {
                    "sa_label": f"text::{token}",
                    "display_text": token,
                    "family": "text",
                    "source_type": "internal_draft_read",
                    "real_energy": 0.18,
                    "anchor_meta": anchor,
                }
            )
        return items

    def _pick_expected_token(
        self,
        *,
        fast_cn: list[dict],
        slow_cn: list[dict],
        exclude_token: str = "",
        visible_text: str = "",
    ) -> str:
        return str(
            self._pick_expected_token_info(
                fast_cn=fast_cn,
                slow_cn=slow_cn,
                exclude_token=exclude_token,
                visible_text=visible_text,
            ).get("token", "")
            or ""
        )

    def _pick_expected_token_info(
        self,
        *,
        fast_cn: list[dict],
        slow_cn: list[dict],
        exclude_token: str = "",
        visible_text: str = "",
    ) -> dict:
        exclude = str(exclude_token or "")
        current_text = str(visible_text or "")
        current_len = len(current_text)
        aligned_best_token = ""
        aligned_best_energy = -1.0
        fallback_best_token = ""
        fallback_best_energy = -1.0
        for branch in list(slow_cn) + list(fast_cn):
            for item in branch.get("predicted_items", []) or []:
                label = str((item or {}).get("sa_label", "") or "")
                if not label.startswith("text::"):
                    continue
                token = label.split("::", 1)[-1]
                if not token or token == exclude:
                    continue
                meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
                alignment = self._prediction_token_alignment(token=token, meta=meta, visible_text=current_text, visible_length=current_len)
                if alignment == "misaligned":
                    continue
                try:
                    energy = float((item or {}).get("virtual_energy", 0.2) or 0.2)
                except (TypeError, ValueError):
                    energy = 0.2
                if alignment == "aligned":
                    if energy > aligned_best_energy:
                        aligned_best_token = token
                        aligned_best_energy = energy
                elif energy > fallback_best_energy:
                    fallback_best_token = token
                    fallback_best_energy = energy
        if aligned_best_token:
            return {"token": aligned_best_token, "alignment": "aligned", "energy": _round4(aligned_best_energy)}
        if fallback_best_token:
            return {"token": fallback_best_token, "alignment": "fallback", "energy": _round4(fallback_best_energy)}
        return {"token": "", "alignment": "", "energy": 0.0}

    def _prediction_token_alignment(self, *, token: str, meta: dict, visible_text: str, visible_length: int) -> str:
        """
        Respect low-grain char-trace process metadata when it exists.

        GL skill imports already record which glyph position a predicted
        `text::x` belongs to. The actuator should not use a later-position
        character as the next token for an earlier draft cursor.
        """

        has_position = any(key in meta for key in ("current_glyph_index", "visible_length", "cursor", "cursor_index", "previous_prefix"))
        if not has_position:
            return "fallback"

        for key in ("visible_length", "current_glyph_index", "cursor", "cursor_index"):
            if key not in meta:
                continue
            try:
                if int(meta.get(key)) != int(visible_length):
                    return "misaligned"
            except (TypeError, ValueError):
                return "misaligned"

        previous_prefix = str(meta.get("previous_prefix", "") or "")
        if previous_prefix and previous_prefix != str(visible_text or ""):
            return "misaligned"

        variant_text = str(meta.get("variant_text", "") or meta.get("expected_text", "") or "")
        if variant_text:
            if not variant_text.startswith(str(visible_text or "")):
                return "misaligned"
            if not variant_text[len(str(visible_text or "")) :].startswith(str(token or "")):
                return "misaligned"

        return "aligned"

    def _latest_mismatch_token(self) -> tuple[int, dict | None]:
        if not self._visible_tokens:
            return -1, None
        for index in range(len(self._visible_tokens) - 1, -1, -1):
            row = self._visible_tokens[index]
            if not isinstance(row, dict):
                continue
            token = str(row.get("token", "") or "")
            event_type = str(row.get("event_type", "") or "")
            if event_type == "write_mismatch":
                return index, row
            if self._is_mismatch_row(row):
                return index, row
        return -1, None

    def _visible_text(self) -> str:
        return "".join(str(entry.get("token", "") or "") for entry in self._visible_tokens if str(entry.get("token", "") or ""))

    def _bounded_cursor(self, value) -> int:
        try:
            cursor = int(value)
        except (TypeError, ValueError):
            cursor = len(self._visible_tokens)
        return max(0, min(len(self._visible_tokens), cursor))

    def _resolve_span(self, span, *, default_to_cursor_previous: bool) -> tuple[int, int]:
        if isinstance(span, dict):
            start = span.get("start", span.get("from", span.get("begin", None)))
            end = span.get("end", span.get("to", None))
        elif isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = span[0], span[1]
        else:
            start = None
            end = None
        if start is None or end is None:
            if default_to_cursor_previous:
                cursor = self._bounded_cursor(self._cursor_index)
                start_i = max(0, cursor - 1)
                end_i = min(len(self._visible_tokens), cursor if self._visible_tokens else 0)
            else:
                start_i = 0
                end_i = len(self._visible_tokens)
            return (start_i, max(start_i, end_i))
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            return self._resolve_span(None, default_to_cursor_previous=default_to_cursor_previous)
        start_i = max(0, min(len(self._visible_tokens), start_i))
        end_i = max(0, min(len(self._visible_tokens), end_i))
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        return (start_i, end_i)

    def _span_text(self, span: tuple[int, int]) -> str:
        return "".join(str(row.get("token", "") or "") for row in self._visible_tokens[span[0] : span[1]])

    def _split_replacement_text(self, text: str) -> list[str]:
        clean = str(text or "")
        if not clean:
            return []
        # TextSensor tokens are often already minimal SA units. For explicit
        # editor actions, each character is a stable edit unit so cursor/span
        # replacement can address the middle of a buffer deterministically.
        return list(clean)
