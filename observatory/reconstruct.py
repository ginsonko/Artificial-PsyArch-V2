from __future__ import annotations

from typing import Callable


SnapshotLookup = Callable[[str], dict | None]
SuccessorLookup = Callable[..., list[dict]]


def reconstruct_memory_detail(
    memory_id: str,
    *,
    snapshot_lookup: SnapshotLookup,
    successor_lookup: SuccessorLookup | None = None,
    successor_top_k: int | None = None,
) -> dict:
    """
    Rebuild detailed white-box memory data outside the realtime tick.

    Summary traces should carry refs. The observatory can call this helper when
    the operator opens a Bn/Cn row and needs the full 1024-level snapshot.
    """

    clean = str(memory_id or "")
    if not clean:
        return {"found": False, "memory_id": ""}
    snapshot = snapshot_lookup(clean)
    if not snapshot:
        return {"found": False, "memory_id": clean}

    items = list(snapshot.get("items", []) or [])
    core_items = list(snapshot.get("core_items", []) or [])
    memory_kind = str(snapshot.get("memory_kind", "") or "")
    successors: list[dict] = []
    if successor_lookup is not None and memory_kind:
        successors = successor_lookup(clean, memory_kind=memory_kind, top_k=successor_top_k)

    return {
        "found": True,
        "memory_id": clean,
        "memory_kind": memory_kind,
        "tick_index": int(snapshot.get("tick_index", -1) or -1),
        "source_text": str(snapshot.get("source_text", "") or ""),
        "item_count": len(items),
        "core_item_count": len(core_items),
        "focus_labels": list(snapshot.get("focus_labels", []) or []),
        "asset_refs": list(snapshot.get("asset_refs", []) or []),
        "sequence_features": dict(snapshot.get("sequence_features", {}) or {}),
        "energy_summary": _energy_summary(core_items or items),
        "items": items,
        "core_items": core_items,
        "prediction_payload_items": list(snapshot.get("prediction_payload_items", []) or []),
        "action_feedback_items": list(snapshot.get("action_feedback_items", []) or []),
        "successors": successors,
    }


def reconstruct_bn_row_detail(
    row: dict,
    *,
    snapshot_lookup: SnapshotLookup,
    successor_lookup: SuccessorLookup | None = None,
    successor_top_k: int | None = None,
) -> dict:
    memory_id = str((row or {}).get("memory_id", "") or "")
    detail = reconstruct_memory_detail(
        memory_id,
        snapshot_lookup=snapshot_lookup,
        successor_lookup=successor_lookup,
        successor_top_k=successor_top_k,
    )
    detail["bn_score"] = float((row or {}).get("score", 0.0) or 0.0)
    detail["score_breakdown"] = dict((row or {}).get("score_breakdown", {}) or {})
    detail["candidate_sources"] = list((row or {}).get("candidate_sources", []) or [])
    detail["matched_tokens"] = dict((row or {}).get("matched_tokens", {}) or {})
    detail["learned_score"] = float((row or {}).get("learned_score", 0.0) or 0.0)
    detail["learned_contributions"] = list((row or {}).get("learned_contributions", []) or [])
    detail["energy_transfer"] = dict((row or {}).get("energy_transfer", {}) or {})
    detail["normalized_weight"] = float((row or {}).get("normalized_weight", 0.0) or 0.0)
    detail["match_efficiency"] = float((row or {}).get("match_efficiency", 0.0) or 0.0)
    detail["b_effective_real_energy"] = float((row or {}).get("b_effective_real_energy", 0.0) or 0.0)
    detail["b_effective_virtual_energy"] = float((row or {}).get("b_effective_virtual_energy", 0.0) or 0.0)
    return detail


def reconstruct_cn_row_detail(
    row: dict,
    *,
    snapshot_lookup: SnapshotLookup,
    successor_lookup: SuccessorLookup | None = None,
    successor_top_k: int | None = None,
) -> dict:
    source_id = str((row or {}).get("source_memory_id", "") or "")
    successor_id = str((row or {}).get("successor_memory_id", "") or "")
    successor_detail = reconstruct_memory_detail(
        successor_id,
        snapshot_lookup=snapshot_lookup,
        successor_lookup=successor_lookup,
        successor_top_k=successor_top_k,
    )
    return {
        "found": bool(successor_detail.get("found", False)),
        "source_memory_id": source_id,
        "successor_memory_id": successor_id,
        "score": float((row or {}).get("score", 0.0) or 0.0),
        "learned_transition_score": float((row or {}).get("learned_transition_score", 0.0) or 0.0),
        "predicted_labels": [str(item.get("sa_label", "") or "") for item in list((row or {}).get("predicted_items", []) or [])[:16]],
        "learned_transition_contributions": list((row or {}).get("learned_transition_contributions", []) or [])[:12],
        "successor": successor_detail,
    }


def reconstruct_tick_observatory(
    trace: dict,
    *,
    snapshot_lookup: SnapshotLookup,
    successor_lookup: SuccessorLookup | None = None,
    max_bn: int = 4,
    max_cn: int = 4,
) -> dict:
    """
    Build an APV2.1-native white-box observatory package after the tick.

    Runtime summary traces intentionally keep refs/light rows. This helper is
    where the heavier expansion belongs.
    """

    fast_bn = list(((trace or {}).get("fast_system", {}) or {}).get("bn", []) or [])
    fast_cn = list(((trace or {}).get("fast_system", {}) or {}).get("cn", []) or [])
    slow_bn = list(((trace or {}).get("slow_system", {}) or {}).get("bn_prime", []) or [])
    slow_cn = list(((trace or {}).get("slow_system", {}) or {}).get("cn_prime", []) or [])
    state_items = list((((trace or {}).get("state_pool", {}) or {}).get("snapshot", {}) or {}).get("items", []) or [])
    explainability = dict((trace or {}).get("explainability", {}) or {})
    action = dict((trace or {}).get("action", {}) or {})
    multimodal = dict((trace or {}).get("multimodal", {}) or {})
    thought_view = dict((trace or {}).get("thought_view", {}) or {})
    expectation_pressure = dict((trace or {}).get("expectation_pressure", {}) or {})
    fast_bn_detail = [
        reconstruct_bn_row_detail(row, snapshot_lookup=snapshot_lookup, successor_lookup=successor_lookup, successor_top_k=max_cn)
        for row in fast_bn[: max(0, int(max_bn))]
    ]
    slow_bn_detail = [
        reconstruct_bn_row_detail(row, snapshot_lookup=snapshot_lookup, successor_lookup=successor_lookup, successor_top_k=max_cn)
        for row in slow_bn[: max(0, int(max_bn))]
    ]
    fast_cn_detail = [
        reconstruct_cn_row_detail(row, snapshot_lookup=snapshot_lookup, successor_lookup=successor_lookup, successor_top_k=max_cn)
        for row in fast_cn[: max(0, int(max_cn))]
    ]
    slow_cn_detail = [
        reconstruct_cn_row_detail(row, snapshot_lookup=snapshot_lookup, successor_lookup=successor_lookup, successor_top_k=max_cn)
        for row in slow_cn[: max(0, int(max_cn))]
    ]
    recalled_details = fast_bn_detail + slow_bn_detail
    prediction_rows = fast_cn + slow_cn
    return {
        "schema_id": "apv21_observatory_reconstruction/v1",
        "tick_index": (trace or {}).get("tick_index"),
        "trace_mode": str((trace or {}).get("trace_mode", "") or ""),
        "state_pool": {
            "energy_summary": _energy_summary(state_items),
            "top_items": _top_energy_items(state_items, limit=12),
        },
        "fast_system": {
            "bn": fast_bn_detail,
            "cn": fast_cn_detail,
        },
        "slow_system": {
            "bn_prime": slow_bn_detail,
            "cn_prime": slow_cn_detail,
            "focus_continuation": dict(((trace or {}).get("slow_system", {}) or {}).get("focus_continuation", {}) or {}),
            "successor_bias": dict(((trace or {}).get("slow_system", {}) or {}).get("successor_bias", {}) or {}),
            "successor_bias_update": dict(((trace or {}).get("slow_system", {}) or {}).get("successor_bias_update", {}) or {}),
        },
        "focus": {
            "selected_labels": list(((trace or {}).get("attention", {}) or {}).get("selected_labels", []) or []),
            "focus_order": dict(((trace or {}).get("attention", {}) or {}).get("focus_order", {}) or {}),
            "ranked_items": list((explainability.get("focus", {}) or {}).get("ranked_items", []) or [])[:12],
            "reason": dict((thought_view.get("focus_reason", {}) or {})),
            "continuation": dict(
                (((trace or {}).get("slow_system", {}) or {}).get("focus_continuation", {}) or {})
                or ((explainability.get("focus", {}) or {}).get("continuation", {}) or {})
            ),
            "successor_bias": dict(((trace or {}).get("slow_system", {}) or {}).get("successor_bias", {}) or {}),
        },
        "feelings": {
            "cognitive": dict(((trace or {}).get("cognitive_feelings", {}) or {}).get("channels", {}) or {}),
            "expectation_pressure": dict(expectation_pressure.get("channels", {}) or {}),
            "expectation_pressure_anchors": reconstruct_expectation_pressure_anchor_view(expectation_pressure, action),
            "emotion": dict(((trace or {}).get("emotion", {}) or {}).get("update", {}) or {}),
        },
        "action": reconstruct_action_view(action, explainability=explainability),
        "inner_world": reconstruct_inner_world(trace, recalled_details=recalled_details, prediction_rows=prediction_rows),
        "learning": {
            "online_embedding": dict(((trace or {}).get("learning", {}) or {}).get("online_embedding", {}) or {}),
            "numeric_channels": _numeric_channel_summary(fast_bn_detail + slow_bn_detail),
        },
        "tuner": dict((trace or {}).get("tuner", {}) or {}),
    }


def reconstruct_expectation_pressure_anchor_view(expectation_pressure_trace: dict, action_trace: dict | None = None) -> dict:
    anchor_trace = dict((expectation_pressure_trace or {}).get("anchor_verification", {}) or {})
    anchors = [dict(anchor) for anchor in list(anchor_trace.get("anchors", []) or []) if isinstance(anchor, dict)]
    verified = [dict(anchor) for anchor in list(anchor_trace.get("verified", []) or []) if isinstance(anchor, dict)]
    missed = [dict(anchor) for anchor in list(anchor_trace.get("missed", []) or []) if isinstance(anchor, dict)]
    created = [dict(anchor) for anchor in list(anchor_trace.get("created", []) or []) if isinstance(anchor, dict)]
    control_rows = [
        dict(item)
        for item in list((action_trace or {}).get("control_items", []) or [])
        if isinstance(item, dict) and str((item.get("anchor_meta", {}) or {}).get("control_kind", "") or "") == "recall_by_expectation"
    ]
    return {
        "schema_id": "expectation_pressure_anchor_observatory/v1",
        "policy": dict(anchor_trace.get("policy", {}) or {}),
        "active_count": int(anchor_trace.get("active_count", len(anchors)) or 0),
        "created_count": len(created),
        "verified_count": len(verified),
        "missed_count": len(missed),
        "active": [_anchor_view_row(anchor) for anchor in anchors[:12]],
        "created": [_anchor_view_row(anchor) for anchor in created[:8]],
        "verified": [_anchor_view_row(anchor) for anchor in verified[:8]],
        "missed": [_anchor_view_row(anchor) for anchor in missed[:8]],
        "recall_control_items": [
            {
                "sa_label": str(row.get("sa_label", "") or ""),
                "virtual_energy": float(row.get("virtual_energy", 0.0) or 0.0),
                "anchor_id": str((row.get("anchor_meta", {}) or {}).get("anchor_id", "") or ""),
                "source_memory_id": str((row.get("anchor_meta", {}) or {}).get("source_memory_id", "") or ""),
                "anchor_type": str((row.get("anchor_meta", {}) or {}).get("anchor_type", "") or ""),
            }
            for row in control_rows[:12]
        ],
    }


def _anchor_view_row(anchor: dict) -> dict:
    return {
        "anchor_id": str((anchor or {}).get("anchor_id", "") or ""),
        "anchor_type": str((anchor or {}).get("anchor_type", "") or ""),
        "source_memory_id": str((anchor or {}).get("source_memory_id", "") or ""),
        "source_memory_kind": str((anchor or {}).get("source_memory_kind", "") or ""),
        "source_tick_index": int((anchor or {}).get("source_tick_index", -1) or -1),
        "level": float((anchor or {}).get("level", 0.0) or 0.0),
        "expected_reward": float((anchor or {}).get("expected_reward", 0.0) or 0.0),
        "expected_punishment": float((anchor or {}).get("expected_punishment", 0.0) or 0.0),
        "expected_pressure": float((anchor or {}).get("expected_pressure", 0.0) or 0.0),
        "verification_state": str((anchor or {}).get("verification_state", "") or ""),
        "age": int((anchor or {}).get("age", 0) or 0),
    }


def reconstruct_action_view(action_trace: dict, *, explainability: dict | None = None) -> dict:
    action_explain = dict((explainability or {}).get("action", {}) or {})
    candidates = list((action_trace or {}).get("candidates", []) or action_explain.get("top_candidates", []) or [])
    selected = list((action_trace or {}).get("selected_actions", []) or action_explain.get("selected_actions", []) or [])
    drive_state = dict((action_trace or {}).get("drive_state", {}) or {})
    outcome_memory = dict(drive_state.get("outcome_memory", {}) or {})
    return {
        "schema_id": "apv21_action_observatory/v1",
        "selected_actions": selected[:8],
        "top_candidates": candidates[:8],
        "drive_state": {
            "bias": dict(drive_state.get("bias", {}) or {}),
            "fatigue": dict(drive_state.get("fatigue", {}) or {}),
            "feedback_modulation": dict(drive_state.get("feedback_modulation", {}) or {}),
            "outcome_memory": outcome_memory,
        },
        "consequence_trace": dict((action_trace or {}).get("consequence_trace", {}) or action_explain.get("consequence_trace", {}) or {}),
        "competition_trace": dict((action_trace or {}).get("competition_trace", {}) or action_explain.get("competition_trace", {}) or {}),
        "causal_window": dict((action_trace or {}).get("causal_window", {}) or action_explain.get("causal_window", {}) or {}),
        "safety_gate": dict((action_trace or {}).get("safety_gate", {}) or action_explain.get("safety_gate", {}) or {}),
        "control_items": list((action_trace or {}).get("control_items", []) or [])[:12],
        "visual_gaze": dict((action_trace or {}).get("visual_gaze", {}) or action_explain.get("visual_gaze", {}) or {}),
    }


def reconstruct_inner_world(trace: dict, *, recalled_details: list[dict] | None = None, prediction_rows: list[dict] | None = None) -> dict:
    multimodal = dict((trace or {}).get("multimodal", {}) or {})
    thought_view = dict((trace or {}).get("thought_view", {}) or {})
    text_output = dict((trace or {}).get("text_output", {}) or {})
    short_term_echo = dict((trace or {}).get("short_term_echo", {}) or {})
    inner_vision = dict(multimodal.get("inner_vision", {}) or {})
    inner_audio = dict(multimodal.get("inner_audio", {}) or {})
    return {
        "schema_id": "apv21_inner_world_view/v2",
        "inner_vision": inner_vision,
        "inner_audio": inner_audio,
        "short_term_echo": reconstruct_short_term_echo_view(short_term_echo),
        "inner_vision_reconstruction": _reconstruct_inner_vision_layers(
            inner_vision,
            recalled_details=recalled_details or [],
            prediction_rows=prediction_rows or [],
            short_term_echo=short_term_echo,
        ),
        "inner_audio_reconstruction": _reconstruct_inner_audio_layers(
            inner_audio,
            recalled_details=recalled_details or [],
            prediction_rows=prediction_rows or [],
            short_term_echo=short_term_echo,
        ),
        "inner_thought": {
            "fast_refs": dict(thought_view.get("fast", {}) or {}),
            "slow_refs": dict(thought_view.get("slow", {}) or {}),
            "feelings": dict(thought_view.get("feelings", {}) or {}),
            "expectation_pressure": dict(thought_view.get("expectation_pressure", {}) or {}),
            "text_output": {
                "visible_text": str(text_output.get("visible_text", "") or (thought_view.get("text_output", {}) or {}).get("visible_text", "") or ""),
                "revision_detected": bool(text_output.get("revision_detected", False) or (thought_view.get("text_output", {}) or {}).get("revision_detected", False)),
            },
        },
    }


def reconstruct_short_term_echo_view(short_term_echo: dict) -> dict:
    items = [dict(item) for item in list((short_term_echo or {}).get("items", []) or []) if isinstance(item, dict)]
    preview = []
    for item in items[:12]:
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        preview.append(
            {
                "sa_label": str(item.get("sa_label", "") or ""),
                "source_type": str(item.get("source_type", "") or ""),
                "family": str(item.get("family", "") or ""),
                "real_energy": _round4(float(item.get("real_energy", 0.0) or 0.0)),
                "echo_kind": str(meta.get("echo_kind", "") or ""),
                "echo_modality": str(meta.get("echo_modality", "") or ""),
                "age_ticks": int(meta.get("age_ticks", 0) or 0),
                "origin_tick_index": int(meta.get("origin_tick_index", -1) or -1),
                "not_new_external_input": bool(meta.get("not_new_external_input", False)),
                "target_label": str(meta.get("echo_target_label", "") or ""),
            }
        )
    return {
        "schema_id": "apv21_short_term_echo_observatory/v1",
        "applied": bool((short_term_echo or {}).get("applied", False)),
        "echo_count": int((short_term_echo or {}).get("echo_count", len(items)) or 0),
        "source_counts": dict((short_term_echo or {}).get("source_counts", {}) or {}),
        "items_preview": preview,
        "policy": str((short_term_echo or {}).get("policy", "") or ""),
        "meaning": "observatory_readout_of_decayed_short_term_echo_not_current_external_input",
    }


def _reconstruct_inner_vision_layers(
    inner_vision: dict,
    *,
    recalled_details: list[dict],
    prediction_rows: list[dict],
    short_term_echo: dict | None = None,
) -> dict:
    assets: list[dict] = []
    frame = dict(inner_vision.get("current_frame", {}) or {})
    field = dict(inner_vision.get("field_reconstruction", {}) or {})
    field_payloads = dict(field.get("payloads", {}) or {})
    objects = []
    for obj in list(inner_vision.get("object_reconstruction", []) or []):
        if not isinstance(obj, dict):
            continue
        salience = max(0.0, float(obj.get("salience", 0.0) or 0.0))
        reconstruction_payload = dict(obj.get("reconstruction_payload", {}) or {})
        sampling_focus = dict(obj.get("sampling_focus", {}) or {})
        row = {
            "source": "real_input",
            "slot": int(obj.get("slot", len(objects)) or len(objects)),
            "bbox_norm": list(obj.get("bbox_norm", []) or []),
            "mean_rgb": list(obj.get("mean_rgb", []) or []),
            "motion_vector": list(obj.get("motion_vector", []) or []),
            "palette": _palette_from_bundle(reconstruction_payload),
            "mask_payload": _payload_from_bundle(reconstruction_payload, "vision.object.mask_grid"),
            "contour_payload": _payload_from_bundle(reconstruction_payload, "vision.object.contour_points"),
            "color_layout_payload": _payload_from_bundle(reconstruction_payload, "vision.object.color_layout"),
            "edge_layout_payload": _payload_from_bundle(reconstruction_payload, "vision.object.edge_layout"),
            "focus_detail_patch_payload": _payload_from_bundle(reconstruction_payload, "vision.object.focus_detail_patch"),
            "reconstruction_payload": reconstruction_payload,
            "opacity": _opacity_from_energy(salience),
            "z_index": int(1000 - min(999, round(salience * 100.0))),
            "salience": round(salience, 4),
            "asset_ref": {},
            "focus_tile_asset_ref": {},
            "reconstruction_basis": "state_pool_numeric_channels",
            "sampling_focus": sampling_focus,
            "focus_precision": _round4(float(sampling_focus.get("precision", 0.0) or 0.0)),
            "variable_resolution": _vision_variable_resolution(
                reconstruction_payload,
                sampling_focus=sampling_focus,
                fallback=dict(obj.get("variable_resolution", {}) or {}),
            ),
        }
        objects.append(row)
    recalled_objects = _recalled_vision_objects(recalled_details)
    recalled_assets: list[dict] = []
    echo_objects = _echo_vision_objects(short_term_echo or {})
    predicted_objects = _predicted_vision_objects(prediction_rows)
    predicted_labels = _predicted_labels(prediction_rows, prefixes=("vision", "vision_obj"))
    layers = []
    if objects:
        layers.append(
            {
                "layer_id": "vision_real_input",
                "layer_type": "real",
                "description": "current visual evidence reconstructed from state-pool numeric channels",
                "opacity_policy": "real_input_salience_to_opacity",
                "object_count": len(objects),
                "asset_count": 0,
                "asset_shortcut_used": False,
                "field_channels": sorted(field_payloads),
            }
        )
    if recalled_objects:
        layers.append(
            {
                "layer_id": "vision_recalled_memory",
                "layer_type": "recalled",
                "description": "B/B' visual memory payloads used as numeric imagination completion",
                "object_count": len(recalled_objects),
                "asset_count": 0,
                "asset_shortcut_used": False,
            }
        )
    if echo_objects:
        layers.append(
            {
                "layer_id": "vision_short_term_echo",
                "layer_type": "short_term_echo",
                "description": "decayed visual afterimage / recent visual residue already present in the state pool",
                "object_count": len(echo_objects),
                "asset_count": 0,
                "asset_shortcut_used": False,
                "not_new_external_input": True,
            }
        )
    if predicted_labels or predicted_objects:
        layers.append(
            {
                "layer_id": "vision_predicted",
                "layer_type": "predicted",
                "description": "C-object visual labels currently carrying virtual prediction energy",
                "object_count": len(predicted_objects),
                "predicted_labels": predicted_labels[:16],
            }
        )
    focus_overlay = _vision_focus_overlay(frame=frame, field_payloads=field_payloads, objects=objects + recalled_objects + echo_objects + predicted_objects)
    return {
        "schema_id": "inner_vision_reconstruction/v2",
        "frame": frame,
        "reconstruction_basis": "state_pool_numeric_channels",
        "asset_shortcut_used": False,
        "field": {
            "schema_id": "vision_field_reconstruction/v1",
            "reconstruction_basis": "state_pool_numeric_channels",
            "payloads": field_payloads,
            "sensor_focus_state": dict(frame.get("sensor_focus_state", {}) or {}),
        },
        "layers": layers,
        "objects": sorted(objects + recalled_objects + echo_objects + predicted_objects, key=lambda row: (int(row.get("z_index", 0) or 0), int(row.get("slot", 0) or 0))),
        "focus_overlay": focus_overlay,
        "assets": [],
        "asset_groups": {},
        "resolver_hints": [],
        "recalled_assets": recalled_assets,
        "predicted_labels": predicted_labels[:24],
        "relations": [],
    }


def _reconstruct_inner_audio_layers(
    inner_audio: dict,
    *,
    recalled_details: list[dict],
    prediction_rows: list[dict],
    short_term_echo: dict | None = None,
) -> dict:
    assets: list[dict] = []
    preview = dict(inner_audio.get("preview_asset_ref", {}) or {})
    feature_summary = dict(inner_audio.get("feature_summary", {}) or {})
    focus = dict(inner_audio.get("focus_reconstruction", {}) or {})
    focus_payloads = dict(focus.get("payloads", {}) or {})
    event_reconstruction = dict(inner_audio.get("event_reconstruction", {}) or {})
    focus_state = dict(
        focus.get("sensor_focus_state", {})
        or preview.get("sensor_focus_state", {})
        or event_reconstruction.get("sensor_focus_state", {})
        or {}
    )
    if not event_reconstruction and focus_payloads:
        event_reconstruction = {
            "schema_id": "reconstruction_payload_bundle/v1",
            "modality": "audio",
            "scope": "focus",
            "sensor_focus_state": focus_state,
            "channels": focus_payloads,
        }
    recalled_assets: list[dict] = []
    recalled_events = _recalled_audio_events(recalled_details)
    echo_events = _echo_audio_events(short_term_echo or {})
    predicted_events = _predicted_audio_events(prediction_rows)
    predicted_labels = _predicted_labels(prediction_rows, prefixes=("audio", "audio_event"))
    layers = []
    if feature_summary:
        layers.append(
            {
                "layer_id": "audio_real_input",
                "layer_type": "real",
                "description": "current audio evidence reconstructed from state-pool numeric channels",
                "asset_count": 0,
                "asset_shortcut_used": False,
                "feature_summary": feature_summary,
                "focus_channels": sorted(focus_payloads),
            }
        )
    if recalled_events:
        layers.append(
            {
                "layer_id": "audio_recalled_memory",
                "layer_type": "recalled",
                "description": "B/B' audio memory payloads used as numeric imagination completion",
                "event_count": len(recalled_events),
                "asset_count": 0,
                "asset_shortcut_used": False,
            }
        )
    if echo_events:
        layers.append(
            {
                "layer_id": "audio_short_term_echo",
                "layer_type": "short_term_echo",
                "description": "decayed aftersound / recent auditory residue already present in the state pool",
                "event_count": len(echo_events),
                "asset_count": 0,
                "asset_shortcut_used": False,
                "not_new_external_input": True,
            }
        )
    if predicted_labels or predicted_events:
        layers.append(
            {
                "layer_id": "audio_predicted",
                "layer_type": "predicted",
                "description": "C-object audio labels currently carrying virtual prediction energy",
                "event_count": len(predicted_events),
                "predicted_labels": predicted_labels[:16],
            }
        )
    focus_band_overlay = _audio_focus_band_overlay(
        focus_state=focus_state,
        focus_payloads=focus_payloads,
        feature_summary=feature_summary,
        preview=preview,
    )
    return {
        "schema_id": "inner_audio_reconstruction/v2",
        "reconstruction_basis": "state_pool_numeric_channels",
        "asset_shortcut_used": False,
        "preview": {
            "preview_duration_ms": float(preview.get("preview_duration_ms", 0.0) or 0.0),
            "sample_rate": int(preview.get("sample_rate", 0) or 0),
            "raw_preview_payload": False,
        },
        "focus": {
            "schema_id": "audio_focus_reconstruction/v1",
            "reconstruction_basis": "state_pool_numeric_channels",
            "sensor_focus_state": focus_state,
            "payloads": focus_payloads,
            "focus_band_overlay": focus_band_overlay,
        },
        "focus_band_overlay": focus_band_overlay,
        "layers": layers,
        "events": ([
            {
                "source": "real_input",
                "event_id": "audio_event::current",
                "opacity": _opacity_from_energy(float(feature_summary.get("rms", 0.0) or 0.0) + float(feature_summary.get("onset_strength", 0.0) or 0.0)),
                "feature_summary": feature_summary,
                "current_bands": list(inner_audio.get("current_bands", []) or []),
                "primary_peaks": list(inner_audio.get("primary_peaks", []) or []),
                "reconstruction_payload": event_reconstruction,
                "waveform_payload": _payload_from_bundle(event_reconstruction, "audio.focus.waveform_slice"),
                "envelope_payload": _payload_from_bundle(event_reconstruction, "audio.focus.envelope"),
                "stft_magnitude_payload": _payload_from_bundle(event_reconstruction, "audio.focus.stft_magnitude"),
                "stft_phase_payload": _payload_from_bundle(event_reconstruction, "audio.focus.stft_phase"),
                "pitch_contour_payload": _payload_from_bundle(event_reconstruction, "audio.focus.pitch_contour"),
                "onset_events_payload": _payload_from_bundle(event_reconstruction, "audio.focus.onset_events"),
                "transient_payload": _payload_from_bundle(event_reconstruction, "audio.focus.transient"),
                "harmonic_noise_payload": _payload_from_bundle(event_reconstruction, "audio.focus.harmonic_noise"),
                "sampling_focus": _audio_sampling_focus(focus_state=focus_state, focus_payloads=focus_payloads),
            }
        ]
        if feature_summary
        else []) + recalled_events + echo_events + predicted_events,
        "assets": [],
        "asset_groups": {},
        "resolver_hints": [],
        "recalled_assets": recalled_assets,
        "predicted_labels": predicted_labels[:24],
    }


def _recalled_vision_objects(details: list[dict], *, limit: int = 6) -> list[dict]:
    rows: list[dict] = []
    for detail in details or []:
        if not isinstance(detail, dict) or not bool(detail.get("found", False)):
            continue
        memory_gain = max(
            0.0,
            float(detail.get("b_effective_real_energy", 0.0) or 0.0)
            + float(detail.get("b_effective_virtual_energy", 0.0) or 0.0) * 0.35,
        )
        if memory_gain <= 0.0:
            memory_gain = max(0.0, float(detail.get("bn_score", 0.0) or 0.0)) * 0.12
        for item in list(detail.get("items", []) or []):
            if len(rows) >= limit:
                return rows
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            payload = dict(item.get("reconstruction_payload", {}) or {})
            if not (label.startswith("vision_obj::") or family == "vision_object"):
                continue
            if not payload.get("channels"):
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            slot = len(rows) + 100
            try:
                if label.startswith("vision_obj::slot_"):
                    slot = 100 + int(label.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                slot = len(rows) + 100
            opacity = min(0.42, _opacity_from_energy(memory_gain) * 0.56)
            rows.append(
                {
                    "source": "recalled_memory",
                    "slot": slot,
                    "bbox_norm": list(anchor_meta.get("bbox_norm", []) or []),
                    "mean_rgb": list(anchor_meta.get("mean_rgb", []) or []),
                    "palette": _palette_from_bundle(payload),
                    "mask_payload": _payload_from_bundle(payload, "vision.object.mask_grid"),
                    "contour_payload": _payload_from_bundle(payload, "vision.object.contour_points"),
                    "color_layout_payload": _payload_from_bundle(payload, "vision.object.color_layout"),
                    "edge_layout_payload": _payload_from_bundle(payload, "vision.object.edge_layout"),
                    "focus_detail_patch_payload": _payload_from_bundle(payload, "vision.object.focus_detail_patch"),
                    "reconstruction_payload": payload,
                    "opacity": round(max(0.12, opacity), 4),
                    "z_index": 760 + len(rows),
                    "salience": round(memory_gain, 4),
                    "asset_ref": {},
                    "focus_tile_asset_ref": {},
                    "reconstruction_basis": "recalled_memory_numeric_payload",
                    "sampling_focus": dict(anchor_meta.get("sampling_focus", {}) or {}),
                    "variable_resolution": _vision_variable_resolution(
                        payload,
                        sampling_focus=dict(anchor_meta.get("sampling_focus", {}) or {}),
                        fallback=dict(anchor_meta.get("variable_resolution", {}) or {}),
                    ),
                    "source_memory_id": str(detail.get("memory_id", "") or ""),
                    "source_memory_tick_index": int(detail.get("tick_index", -1) or -1),
                }
            )
    return rows


def _recalled_audio_events(details: list[dict], *, limit: int = 4) -> list[dict]:
    rows: list[dict] = []
    for detail in details or []:
        if not isinstance(detail, dict) or not bool(detail.get("found", False)):
            continue
        memory_gain = max(
            0.0,
            float(detail.get("b_effective_real_energy", 0.0) or 0.0)
            + float(detail.get("b_effective_virtual_energy", 0.0) or 0.0) * 0.35,
        )
        if memory_gain <= 0.0:
            memory_gain = max(0.0, float(detail.get("bn_score", 0.0) or 0.0)) * 0.1
        for item in list(detail.get("items", []) or []):
            if len(rows) >= limit:
                return rows
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            payload = dict(item.get("reconstruction_payload", {}) or {})
            if not (label.startswith("audio_event::") or family == "audio_event"):
                continue
            if not payload.get("channels"):
                continue
            numeric = dict(item.get("numeric_features", {}) or {}) if isinstance(item.get("numeric_features", {}), dict) else {}
            feature_summary = {
                "dominant_hz": _dominant_hz_from_numeric(numeric),
                "rms": max(0.0, float(item.get("real_energy", 0.0) or 0.0)) * 0.1,
                "onset_strength": 0.0,
            }
            rows.append(
                {
                    "source": "recalled_memory",
                    "event_id": label,
                    "opacity": round(max(0.1, min(0.4, _opacity_from_energy(memory_gain) * 0.5)), 4),
                    "feature_summary": feature_summary,
                    "current_bands": [],
                    "primary_peaks": [label],
                    "reconstruction_payload": payload,
                    "waveform_payload": _payload_from_bundle(payload, "audio.focus.waveform_slice"),
                    "envelope_payload": _payload_from_bundle(payload, "audio.focus.envelope"),
                    "stft_magnitude_payload": _payload_from_bundle(payload, "audio.focus.stft_magnitude"),
                    "stft_phase_payload": _payload_from_bundle(payload, "audio.focus.stft_phase"),
                    "pitch_contour_payload": _payload_from_bundle(payload, "audio.focus.pitch_contour"),
                    "onset_events_payload": _payload_from_bundle(payload, "audio.focus.onset_events"),
                    "transient_payload": _payload_from_bundle(payload, "audio.focus.transient"),
                    "harmonic_noise_payload": _payload_from_bundle(payload, "audio.focus.harmonic_noise"),
                    "source_memory_id": str(detail.get("memory_id", "") or ""),
                    "source_memory_tick_index": int(detail.get("tick_index", -1) or -1),
                }
            )
    return rows


def _echo_vision_objects(short_term_echo: dict, *, limit: int = 6) -> list[dict]:
    """
    Rebuild visual afterimage rows from already-applied echo items.

    This is observatory-only: the echo item has already participated in the
    state pool before recall. Here we only expose its numeric payload as a
    separate layer so the operator can see "recent residue" apart from current
    input, B memory completion, and C* prediction.
    """

    rows: list[dict] = []
    for item in _echo_copy_items(short_term_echo, modality="vision"):
        if len(rows) >= limit:
            break
        label = str(item.get("sa_label", "") or "")
        family = str(item.get("family", "") or "")
        payload = dict(item.get("reconstruction_payload", {}) or {})
        if not (label.startswith("vision_obj::") or family == "vision_object"):
            continue
        if not payload.get("channels"):
            continue
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        echo_energy = max(0.0, float(item.get("real_energy", meta.get("echo_energy", 0.0)) or 0.0))
        sampling_focus = dict(meta.get("sampling_focus", {}) or {})
        rows.append(
            {
                "source": "short_term_echo",
                "slot": 300 + len(rows),
                "bbox_norm": list(meta.get("bbox_norm", []) or []),
                "mean_rgb": list(meta.get("mean_rgb", []) or []),
                "palette": _palette_from_bundle(payload),
                "mask_payload": _payload_from_bundle(payload, "vision.object.mask_grid"),
                "contour_payload": _payload_from_bundle(payload, "vision.object.contour_points"),
                "color_layout_payload": _payload_from_bundle(payload, "vision.object.color_layout"),
                "edge_layout_payload": _payload_from_bundle(payload, "vision.object.edge_layout"),
                "focus_detail_patch_payload": _payload_from_bundle(payload, "vision.object.focus_detail_patch"),
                "reconstruction_payload": payload,
                "opacity": round(max(0.08, min(0.32, _opacity_from_energy(echo_energy) * 0.42)), 4),
                "z_index": 820 + len(rows),
                "salience": round(echo_energy, 4),
                "asset_ref": {},
                "focus_tile_asset_ref": {},
                "reconstruction_basis": "short_term_echo_state_pool_numeric_payload",
                "sampling_focus": sampling_focus,
                "focus_precision": _round4(float(sampling_focus.get("precision", 0.0) or 0.0)),
                "variable_resolution": _vision_variable_resolution(
                    payload,
                    sampling_focus=sampling_focus,
                    fallback=dict(meta.get("variable_resolution", {}) or {}),
                ),
                "echo_kind": str(meta.get("echo_kind", "") or ""),
                "echo_modality": str(meta.get("echo_modality", "") or ""),
                "age_ticks": int(meta.get("age_ticks", 0) or 0),
                "origin_tick_index": int(meta.get("origin_tick_index", -1) or -1),
                "not_new_external_input": bool(meta.get("not_new_external_input", False)),
                "echo_target_label": str(meta.get("echo_target_label", label) or label),
            }
        )
    return rows


def _echo_audio_events(short_term_echo: dict, *, limit: int = 4) -> list[dict]:
    rows: list[dict] = []
    for item in _echo_copy_items(short_term_echo, modality="audio"):
        if len(rows) >= limit:
            break
        label = str(item.get("sa_label", "") or "")
        family = str(item.get("family", "") or "")
        payload = dict(item.get("reconstruction_payload", {}) or {})
        if not (label.startswith("audio_event::") or family == "audio_event"):
            continue
        if not payload.get("channels"):
            continue
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        numeric = dict(item.get("numeric_features", {}) or {}) if isinstance(item.get("numeric_features", {}), dict) else {}
        echo_energy = max(0.0, float(item.get("real_energy", meta.get("echo_energy", 0.0)) or 0.0))
        feature_summary = {
            "dominant_hz": float(meta.get("dominant_hz", 0.0) or _dominant_hz_from_numeric(numeric)),
            "rms": max(0.0, echo_energy) * 0.12,
            "onset_strength": 0.0,
        }
        rows.append(
            {
                "source": "short_term_echo",
                "event_id": label,
                "opacity": round(max(0.08, min(0.34, _opacity_from_energy(echo_energy) * 0.44)), 4),
                "salience": round(echo_energy, 4),
                "feature_summary": feature_summary,
                "current_bands": [],
                "primary_peaks": [label],
                "reconstruction_payload": payload,
                "waveform_payload": _payload_from_bundle(payload, "audio.focus.waveform_slice"),
                "envelope_payload": _payload_from_bundle(payload, "audio.focus.envelope"),
                "stft_magnitude_payload": _payload_from_bundle(payload, "audio.focus.stft_magnitude"),
                "stft_phase_payload": _payload_from_bundle(payload, "audio.focus.stft_phase"),
                "pitch_contour_payload": _payload_from_bundle(payload, "audio.focus.pitch_contour"),
                "onset_events_payload": _payload_from_bundle(payload, "audio.focus.onset_events"),
                "transient_payload": _payload_from_bundle(payload, "audio.focus.transient"),
                "harmonic_noise_payload": _payload_from_bundle(payload, "audio.focus.harmonic_noise"),
                "reconstruction_basis": "short_term_echo_state_pool_numeric_payload",
                "echo_kind": str(meta.get("echo_kind", "") or ""),
                "echo_modality": str(meta.get("echo_modality", "") or ""),
                "age_ticks": int(meta.get("age_ticks", 0) or 0),
                "origin_tick_index": int(meta.get("origin_tick_index", -1) or -1),
                "not_new_external_input": bool(meta.get("not_new_external_input", False)),
                "echo_target_label": str(meta.get("echo_target_label", label) or label),
            }
        )
    return rows


def _echo_copy_items(short_term_echo: dict, *, modality: str) -> list[dict]:
    rows: list[dict] = []
    for item in list((short_term_echo or {}).get("items", []) or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("sa_label", "") or "")
        if label.startswith("echo::"):
            continue
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        if not bool(meta.get("is_echo", False)):
            continue
        if str(meta.get("echo_modality", "") or "") != str(modality):
            continue
        if not bool(meta.get("not_new_external_input", False)):
            continue
        rows.append(dict(item))
    rows.sort(
        key=lambda row: (
            -float(row.get("real_energy", 0.0) or 0.0),
            int((row.get("anchor_meta", {}) or {}).get("age_ticks", 999) or 999),
            str(row.get("sa_label", "") or ""),
        )
    )
    return rows


def _predicted_vision_objects(rows: list[dict], *, limit: int = 6) -> list[dict]:
    predicted: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for item in list(row.get("predicted_items", []) or []):
            if len(predicted) >= limit:
                return predicted
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            payload = dict(item.get("reconstruction_payload", {}) or {})
            if not (label.startswith("vision_obj::") or family == "vision_object"):
                continue
            if not payload.get("channels"):
                continue
            energy = max(0.0, float(item.get("virtual_energy", item.get("real_energy", 0.0)) or 0.0))
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            sampling_focus = dict(anchor_meta.get("sampling_focus", {}) or {})
            predicted.append(
                {
                    "source": "cstar_prediction",
                    "slot": 200 + len(predicted),
                    "bbox_norm": list(anchor_meta.get("bbox_norm", []) or []),
                    "mean_rgb": list(anchor_meta.get("mean_rgb", []) or []),
                    "palette": _palette_from_bundle(payload),
                    "mask_payload": _payload_from_bundle(payload, "vision.object.mask_grid"),
                    "contour_payload": _payload_from_bundle(payload, "vision.object.contour_points"),
                    "color_layout_payload": _payload_from_bundle(payload, "vision.object.color_layout"),
                    "edge_layout_payload": _payload_from_bundle(payload, "vision.object.edge_layout"),
                    "focus_detail_patch_payload": _payload_from_bundle(payload, "vision.object.focus_detail_patch"),
                    "reconstruction_payload": payload,
                    "opacity": round(max(0.1, min(0.34, _opacity_from_energy(energy) * 0.38)), 4),
                    "z_index": 900 + len(predicted),
                    "salience": round(energy, 4),
                    "virtual_energy": round(energy, 4),
                    "asset_ref": {},
                    "focus_tile_asset_ref": {},
                    "reconstruction_basis": "cstar_numeric_prediction_payload",
                    "sampling_focus": sampling_focus,
                    "focus_precision": _round4(float(sampling_focus.get("precision", 0.0) or 0.0)),
                    "variable_resolution": _vision_variable_resolution(
                        payload,
                        sampling_focus=sampling_focus,
                        fallback=dict(anchor_meta.get("variable_resolution", {}) or {}),
                    ),
                    "predicted_label": label,
                }
            )
    return predicted


def _predicted_audio_events(rows: list[dict], *, limit: int = 4) -> list[dict]:
    predicted: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for item in list(row.get("predicted_items", []) or []):
            if len(predicted) >= limit:
                return predicted
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            payload = dict(item.get("reconstruction_payload", {}) or {})
            if not (label.startswith("audio_event::") or family == "audio_event"):
                continue
            if not payload.get("channels"):
                continue
            energy = max(0.0, float(item.get("virtual_energy", item.get("real_energy", 0.0)) or 0.0))
            numeric = dict(item.get("numeric_features", {}) or {}) if isinstance(item.get("numeric_features", {}), dict) else {}
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            feature_summary = {
                "dominant_hz": float(anchor_meta.get("dominant_hz", 0.0) or _dominant_hz_from_numeric(numeric)),
                "rms": max(0.0, energy) * 0.1,
                "onset_strength": 0.0,
            }
            predicted.append(
                {
                    "source": "cstar_prediction",
                    "event_id": label,
                    "opacity": round(max(0.1, min(0.34, _opacity_from_energy(energy) * 0.38)), 4),
                    "feature_summary": feature_summary,
                    "current_bands": [],
                    "primary_peaks": [label],
                    "reconstruction_payload": payload,
                    "waveform_payload": _payload_from_bundle(payload, "audio.focus.waveform_slice"),
                    "envelope_payload": _payload_from_bundle(payload, "audio.focus.envelope"),
                    "stft_magnitude_payload": _payload_from_bundle(payload, "audio.focus.stft_magnitude"),
                    "stft_phase_payload": _payload_from_bundle(payload, "audio.focus.stft_phase"),
                    "pitch_contour_payload": _payload_from_bundle(payload, "audio.focus.pitch_contour"),
                    "onset_events_payload": _payload_from_bundle(payload, "audio.focus.onset_events"),
                    "transient_payload": _payload_from_bundle(payload, "audio.focus.transient"),
                    "harmonic_noise_payload": _payload_from_bundle(payload, "audio.focus.harmonic_noise"),
                    "virtual_energy": round(energy, 4),
                    "predicted_label": label,
                }
            )
    return predicted


def _vision_focus_overlay(*, frame: dict, field_payloads: dict, objects: list[dict]) -> dict:
    focus = dict(frame.get("sensor_focus_state", {}) or {})
    precision_payload = dict((field_payloads or {}).get("vision.field.sampling_precision", {}) or {})
    precision_values = _payload_values(precision_payload)
    scale = _clamp(_float_default(focus.get("scale"), 1.0), 0.35, 1.8)
    center_x = _clamp(_float_default(focus.get("center_x"), 0.5), 0.0, 1.0)
    center_y = _clamp(_float_default(focus.get("center_y"), 0.5), 0.0, 1.0)
    radius = _clamp(0.42 * scale, 0.12, 0.76)
    clarity = _precision_clarity(
        values=precision_values,
        shape=list(precision_payload.get("payload_shape", []) or []),
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    object_focus = []
    resolution_rows = []
    for obj in objects or []:
        if not isinstance(obj, dict):
            continue
        sampling_focus = dict(obj.get("sampling_focus", {}) or {})
        variable_resolution = dict(obj.get("variable_resolution", {}) or {})
        if variable_resolution:
            resolution_rows.append(variable_resolution)
        if sampling_focus:
            object_focus.append(
                {
                    "slot": int(obj.get("slot", len(object_focus)) or len(object_focus)),
                    "source": str(obj.get("source", "") or ""),
                    "bbox_norm": list(obj.get("bbox_norm", []) or []),
                    "distance": _round4(float(sampling_focus.get("distance", 0.0) or 0.0)),
                    "gain": _round4(float(sampling_focus.get("gain", 0.0) or 0.0)),
                    "precision": _round4(float(sampling_focus.get("precision", 0.0) or 0.0)),
                    "focus_radius": _round4(float(sampling_focus.get("focus_radius", radius) or radius)),
                    "variable_resolution": variable_resolution,
                }
            )
    return {
        "schema_id": "vision_focus_overlay/v1",
        "visible": bool(focus or precision_values or object_focus),
        "source": "sensor_focus_state_and_sampling_precision",
        "center_norm": [_round4(center_x), _round4(center_y)],
        "scale": _round4(scale),
        "radius_norm": _round4(radius),
        "sensor_focus_state": focus,
        "precision_grid": {
            "schema_id": "vision_focus_precision_grid/v1",
            "channel": "vision.field.sampling_precision",
            "payload_shape": list(precision_payload.get("payload_shape", []) or []),
            "payload_values": precision_values,
            "sampling_precision": _round4(float(precision_payload.get("sampling_precision", 0.0) or 0.0)),
        },
        "clarity": clarity,
        "object_focus": object_focus[:12],
        "resolution_summary": _resolution_summary(resolution_rows),
    }


def _precision_clarity(*, values: list[float], shape: list, center_x: float, center_y: float, radius: float) -> dict:
    if len(shape) < 2 or not values:
        return {
            "near_focus": 0.0,
            "far_periphery": 0.0,
            "contrast": 0.0,
            "metric_basis": "vision.field.sampling_precision",
        }
    rows = max(1, int(shape[0] or 1))
    cols = max(1, int(shape[1] or 1))
    near: list[float] = []
    far: list[float] = []
    for y in range(rows):
        for x in range(cols):
            idx = y * cols + x
            if idx >= len(values):
                continue
            nx = (x + 0.5) / max(1.0, float(cols))
            ny = (y + 0.5) / max(1.0, float(rows))
            distance = ((nx - center_x) ** 2 + (ny - center_y) ** 2) ** 0.5
            value = float(values[idx])
            if distance <= radius * 0.55:
                near.append(value)
            elif distance >= radius:
                far.append(value)
    near_avg = sum(near) / max(1, len(near))
    far_avg = sum(far) / max(1, len(far))
    return {
        "near_focus": _round4(near_avg),
        "far_periphery": _round4(far_avg),
        "contrast": _round4(near_avg - far_avg),
        "near_cell_count": len(near),
        "far_cell_count": len(far),
        "metric_basis": "vision.field.sampling_precision",
    }


def _audio_focus_band_overlay(*, focus_state: dict, focus_payloads: dict, feature_summary: dict, preview: dict) -> dict:
    focus = dict(focus_state or {})
    sample_rate = max(
        8000,
        int((feature_summary or {}).get("sample_rate", 0) or (preview or {}).get("sample_rate", 0) or 16000),
    )
    nyquist = max(1.0, float(sample_rate) * 0.5)
    center_hz = _clamp(float(focus.get("center_hz", 1000.0) or 1000.0), 40.0, nyquist)
    width_hz = _clamp(float(focus.get("width_hz", 2400.0) or 2400.0), 120.0, nyquist * 2.0)
    low = max(0.0, center_hz - width_hz * 0.5)
    high = min(nyquist, center_hz + width_hz * 0.5)
    precision = _audio_focus_precision(focus_payloads=focus_payloads, sample_rate=sample_rate, width_hz=width_hz)
    return {
        "schema_id": "audio_focus_band_overlay/v1",
        "visible": bool(focus or focus_payloads),
        "source": "sensor_focus_state_and_focus_payload_sampling_precision",
        "center_hz": _round4(center_hz),
        "width_hz": _round4(width_hz),
        "low_hz": _round4(low),
        "high_hz": _round4(high),
        "sample_rate": sample_rate,
        "precision": _round4(precision),
        "center_norm": _round4(center_hz / nyquist),
        "low_norm": _round4(low / nyquist),
        "high_norm": _round4(high / nyquist),
        "sensor_focus_state": focus,
    }


def _audio_sampling_focus(*, focus_state: dict, focus_payloads: dict) -> dict:
    overlay = _audio_focus_band_overlay(
        focus_state=focus_state,
        focus_payloads=focus_payloads,
        feature_summary={},
        preview={},
    )
    return {
        "schema_id": "audio_focused_band_sampling/v1",
        "center_hz": overlay["center_hz"],
        "width_hz": overlay["width_hz"],
        "precision": overlay["precision"],
    }


def _audio_focus_precision(*, focus_payloads: dict, sample_rate: int, width_hz: float) -> float:
    precisions = [
        float(payload.get("sampling_precision", 0.0) or 0.0)
        for payload in (focus_payloads or {}).values()
        if isinstance(payload, dict) and float(payload.get("sampling_precision", 0.0) or 0.0) > 0.0
    ]
    if precisions:
        return max(0.0, min(1.0, sum(precisions) / len(precisions)))
    nyquist = max(1.0, float(sample_rate) * 0.5)
    width_ratio = max(0.02, min(1.0, float(width_hz or 2400.0) / nyquist))
    return max(0.45, min(1.0, 1.05 - width_ratio * 0.45))


def _vision_variable_resolution(bundle: dict, *, sampling_focus: dict, fallback: dict | None = None) -> dict:
    fallback = dict(fallback or {})
    color = _payload_from_bundle(bundle, "vision.object.color_layout")
    mask = _payload_from_bundle(bundle, "vision.object.mask_grid")
    edge = _payload_from_bundle(bundle, "vision.object.edge_layout")
    patch = _payload_from_bundle(bundle, "vision.object.focus_detail_patch")
    color_shape = list(color.get("payload_shape", []) or fallback.get("color_grid_shape", []) or [])
    mask_shape = list(mask.get("payload_shape", []) or fallback.get("mask_grid_shape", []) or [])
    edge_shape = list(edge.get("payload_shape", []) or fallback.get("edge_grid_shape", []) or [])
    patch_shape = list(patch.get("payload_shape", []) or fallback.get("focus_detail_patch_shape", []) or [])
    focus_precision = float(
        sampling_focus.get("precision", 0.0)
        or color.get("sampling_precision", 0.0)
        or fallback.get("focus_precision", 0.0)
        or 0.0
    )
    return {
        "schema_id": "vision_foveated_payload_resolution/v1",
        "policy": str(fallback.get("policy", "") or "continuous_foveated_object_payload_resolution"),
        "tier": str(fallback.get("tier", "") or _resolution_tier(color_shape)),
        "focus_precision": _round4(focus_precision),
        "color_grid_shape": [int(value) for value in color_shape],
        "mask_grid_shape": [int(value) for value in mask_shape],
        "edge_grid_shape": [int(value) for value in edge_shape],
        "focus_detail_patch_shape": [int(value) for value in patch_shape],
        "near_original_focus_patch": bool(patch_shape),
    }


def _resolution_tier(color_shape: list) -> str:
    rows = int(color_shape[0]) if len(color_shape) >= 1 else 0
    if rows >= 20:
        return "focus_high"
    if rows >= 12:
        return "focus_mid"
    if rows > 0:
        return "peripheral_low"
    return "unknown"


def _resolution_summary(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    max_color_cells = 0
    min_color_cells = 0
    focus_patch_count = 0
    max_focus_patch_cells = 0
    policies: dict[str, int] = {}
    cells: list[int] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        policy = str(row.get("policy", "") or "unknown")
        policies[policy] = policies.get(policy, 0) + 1
        tier = str(row.get("tier", "") or "unknown")
        counts[tier] = counts.get(tier, 0) + 1
        shape = list(row.get("color_grid_shape", []) or [])
        if len(shape) >= 2:
            cell_count = int(shape[0] or 0) * int(shape[1] or 0)
            cells.append(cell_count)
        patch_shape = list(row.get("focus_detail_patch_shape", []) or [])
        if len(patch_shape) >= 2:
            patch_cells = int(patch_shape[0] or 0) * int(patch_shape[1] or 0)
            if patch_cells > 0:
                focus_patch_count += 1
                max_focus_patch_cells = max(max_focus_patch_cells, patch_cells)
    if cells:
        max_color_cells = max(cells)
        min_color_cells = min(cells)
    policy = "continuous_foveated_object_payload_resolution" if policies.get("continuous_foveated_object_payload_resolution", 0) else "foveated_object_payload_resolution"
    return {
        "schema_id": "vision_resolution_summary/v1",
        "policy": policy,
        "policy_counts": policies,
        "tier_counts": counts,
        "max_color_cells": max_color_cells,
        "min_color_cells": min_color_cells,
        "focus_detail_patch_count": focus_patch_count,
        "max_focus_patch_cells": max_focus_patch_cells,
        "variable_resolution_active": max_color_cells > min_color_cells or counts.get("focus_high", 0) > 0 or focus_patch_count > 0,
    }


def _dominant_hz_from_numeric(numeric: dict) -> float:
    pitch = list(numeric.get("audio.pitch", []) or [])
    if not pitch:
        return 0.0
    return round(float(pitch[0] or 0.0) * 8000.0, 4)


def _assets_from_recalled_details(details: list[dict], *, modality: str) -> list[dict]:
    refs = []
    for detail in details or []:
        if not isinstance(detail, dict) or not bool(detail.get("found", False)):
            continue
        for ref in list(detail.get("asset_refs", []) or []):
            if isinstance(ref, dict) and str(ref.get("modality", "") or "") == modality:
                row = dict(ref)
                row["reconstruction_role"] = "memory_completion"
                row["source_memory_id"] = str(detail.get("memory_id", "") or "")
                row["source_memory_tick_index"] = int(detail.get("tick_index", -1) or -1)
                refs.append(row)
    return _dedupe_asset_refs(refs)


def _predicted_labels(rows: list[dict], *, prefixes: tuple[str, ...]) -> list[str]:
    labels = []
    seen = set()
    for row in rows or []:
        for item in list((row or {}).get("predicted_items", []) or []):
            label = str((item or {}).get("sa_label", "") or "")
            if not label or label in seen:
                continue
            if any(label.startswith(prefix) for prefix in prefixes):
                seen.add(label)
                labels.append(label)
    return labels


def _dedupe_asset_refs(refs: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        asset_id = str(ref.get("asset_id", "") or "")
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        rows.append(dict(ref))
    return rows


def _group_assets_by_type(refs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for ref in _dedupe_asset_refs(refs):
        asset_type = str(ref.get("asset_type", "") or "unknown_asset")
        groups.setdefault(asset_type, []).append(ref)
    return groups


def _asset_resolver_hints(refs: list[dict]) -> list[dict]:
    hints = []
    for ref in _dedupe_asset_refs(refs):
        payload_ref = dict(ref.get("payload_ref", {}) or {})
        encoding = str(payload_ref.get("encoding", "") or "")
        hints.append(
            {
                "asset_id": str(ref.get("asset_id", "") or ""),
                "asset_type": str(ref.get("asset_type", "") or ""),
                "modality": str(ref.get("modality", "") or ""),
                "encoding": encoding,
                "storage_tier": str(payload_ref.get("storage_tier", "") or ""),
                "byte_length": int(payload_ref.get("byte_length", 0) or 0),
                "can_decode_inline": encoding in {"png", "jpg", "jpeg", "webp", "wav", "json"},
                "has_disk_payload": bool(str(payload_ref.get("path", "") or "")),
                "fidelity_level": str(ref.get("fidelity_level", "") or ""),
                "scope": str(ref.get("scope", "") or ""),
            }
        )
    return hints


def _opacity_from_energy(value: float) -> float:
    energy = max(0.0, float(value or 0.0))
    return round(min(0.96, max(0.22, 0.22 + energy * 0.58)), 4)


def _round4(value: float) -> float:
    return round(float(value), 4)


def _float_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _payload_from_bundle(bundle: dict, channel: str) -> dict:
    if not isinstance(bundle, dict):
        return {}
    channels = dict(bundle.get("channels", {}) or {})
    payload = channels.get(str(channel or ""))
    return dict(payload) if isinstance(payload, dict) else {}


def _payload_values(payload: dict) -> list[float]:
    if not isinstance(payload, dict):
        return []
    values = payload.get("payload_values", [])
    if not isinstance(values, (list, tuple)):
        return []
    rows = []
    for value in values:
        try:
            rows.append(float(value))
        except (TypeError, ValueError):
            rows.append(0.0)
    return rows


def _palette_from_bundle(bundle: dict) -> list[dict]:
    payload = _payload_from_bundle(bundle, "vision.object.palette")
    values = list(payload.get("payload_values", []) or [])
    rows = []
    for idx in range(0, len(values), 4):
        chunk = values[idx : idx + 4]
        if len(chunk) < 4:
            continue
        rows.append(
            {
                "rgb": [float(chunk[0] or 0.0), float(chunk[1] or 0.0), float(chunk[2] or 0.0)],
                "weight": float(chunk[3] or 0.0),
            }
        )
    rows.sort(key=lambda row: -float(row.get("weight", 0.0) or 0.0))
    return rows[:6]


def _energy_summary(items: list[dict]) -> dict:
    total_real = 0.0
    total_virtual = 0.0
    total_pressure = 0.0
    positive_real = 0
    positive_virtual = 0
    top_pressure: list[dict] = []
    for item in items or []:
        real = float((item or {}).get("real_energy", 0.0) or 0.0)
        virtual = float((item or {}).get("virtual_energy", 0.0) or 0.0)
        pressure = float((item or {}).get("cognitive_pressure", real - virtual) or 0.0)
        total_real += real
        total_virtual += virtual
        total_pressure += pressure
        if real > 0.0:
            positive_real += 1
        if virtual > 0.0:
            positive_virtual += 1
        top_pressure.append(
            {
                "sa_label": str((item or {}).get("sa_label", "") or ""),
                "real_energy": real,
                "virtual_energy": virtual,
                "cognitive_pressure": pressure,
            }
        )
    top_pressure.sort(key=lambda row: (-float(row.get("cognitive_pressure", 0.0) or 0.0), str(row.get("sa_label", "") or "")))
    return {
        "total_real_energy": round(total_real, 4),
        "total_virtual_energy": round(total_virtual, 4),
        "total_cognitive_pressure": round(total_pressure, 4),
        "positive_real_count": positive_real,
        "positive_virtual_count": positive_virtual,
        "top_pressure": top_pressure[:12],
    }


def _top_energy_items(items: list[dict], *, limit: int) -> list[dict]:
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        real = float(item.get("real_energy", 0.0) or 0.0)
        virtual = float(item.get("virtual_energy", 0.0) or 0.0)
        pressure = float(item.get("cognitive_pressure", real - virtual) or 0.0)
        rows.append(
            {
                "sa_label": str(item.get("sa_label", "") or ""),
                "family": str(item.get("family", "") or ""),
                "source_type": str(item.get("source_type", "") or ""),
                "real_energy": real,
                "virtual_energy": virtual,
                "cognitive_pressure": pressure,
                "numeric_channels": sorted((item.get("numeric_features", {}) or {}).keys())[:8]
                if isinstance(item.get("numeric_features", {}), dict)
                else [],
            }
        )
    rows.sort(
        key=lambda row: (
            -(float(row.get("real_energy", 0.0) or 0.0) + float(row.get("virtual_energy", 0.0) or 0.0)),
            -float(row.get("cognitive_pressure", 0.0) or 0.0),
            str(row.get("sa_label", "") or ""),
        )
    )
    return rows[: max(1, int(limit))]


def _numeric_channel_summary(bn_details: list[dict]) -> dict:
    channels: dict[str, dict] = {}
    for detail in bn_details or []:
        breakdown = dict(detail.get("score_breakdown", {}) or {})
        numeric = breakdown.get("numeric_channels", {})
        if isinstance(numeric, dict):
            iterable = numeric.items()
        else:
            iterable = []
        for channel, value in iterable:
            clean = str(channel or "")
            if not clean:
                continue
            bucket = channels.setdefault(clean, {"count": 0, "score_sum": 0.0, "max_score": 0.0})
            score = float(value or 0.0)
            bucket["count"] += 1
            bucket["score_sum"] += score
            bucket["max_score"] = max(float(bucket["max_score"]), score)
        for row in list(detail.get("items", []) or [])[:32]:
            if not isinstance(row, dict):
                continue
            numeric_features = row.get("numeric_features", {})
            if not isinstance(numeric_features, dict):
                continue
            for channel in numeric_features:
                clean = str(channel or "")
                if not clean:
                    continue
                bucket = channels.setdefault(clean, {"count": 0, "score_sum": 0.0, "max_score": 0.0})
                bucket["count"] += 1
    return {
        channel: {
            "count": int(value.get("count", 0) or 0),
            "score_sum": round(float(value.get("score_sum", 0.0) or 0.0), 4),
            "max_score": round(float(value.get("max_score", 0.0) or 0.0), 4),
        }
        for channel, value in sorted(channels.items())
    }
