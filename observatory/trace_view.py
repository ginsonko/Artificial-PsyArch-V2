from __future__ import annotations


def summarize_tick(trace: dict) -> dict:
    fast_bn = trace.get("fast_system", {}).get("bn", [])
    fast_cn = trace.get("fast_system", {}).get("cn", [])
    slow_bn = trace.get("slow_system", {}).get("bn_prime", [])
    slow_cn = trace.get("slow_system", {}).get("cn_prime", [])
    state_rows = trace.get("state_pool", {}).get("snapshot", {}).get("items", [])
    multimodal = trace.get("multimodal", {})
    thought_view = trace.get("thought_view", {})
    explainability = trace.get("explainability", {})
    expectation_pressure = trace.get("expectation_pressure", {})
    return {
        "tick_index": trace.get("tick_index"),
        "input_text": trace.get("input", {}).get("normalized_text", ""),
        "top_state": [
            {
                "sa_label": row.get("sa_label"),
                "real_energy": row.get("real_energy"),
                "virtual_energy": row.get("virtual_energy"),
                "cognitive_pressure": row.get("cognitive_pressure"),
            }
            for row in state_rows[:5]
        ],
        "fast_bn_ids": [row.get("memory_id") for row in fast_bn],
        "fast_cn_labels": [item.get("sa_label") for branch in fast_cn for item in branch.get("predicted_items", [])[:2]],
        "slow_bn_ids": [row.get("memory_id") for row in slow_bn],
        "slow_cn_labels": [item.get("sa_label") for branch in slow_cn for item in branch.get("predicted_items", [])[:2]],
        "focus_labels": trace.get("attention", {}).get("selected_labels", []),
        "inner_vision_focus": list((multimodal.get("inner_vision", {}) or {}).get("focus_objects", []) or []),
        "inner_audio_peaks": list((multimodal.get("inner_audio", {}) or {}).get("primary_peaks", []) or []),
        "thought_feelings": dict((thought_view.get("feelings", {}) or {}).get("channels", {}) or {}),
        "thought_expectation_pressure": dict((expectation_pressure.get("channels", {}) or {})),
        "top_bn_reasons": list((explainability.get("fast_bn", []) or [])[:2]),
        "top_action_reasons": list(((explainability.get("action", {}) or {}).get("selected_actions", []) or [])[:2]),
    }
