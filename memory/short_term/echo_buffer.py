from __future__ import annotations

from collections import Counter, deque


def _round4(value: float) -> float:
    return round(float(value), 4)


def _energy_of(item: dict) -> float:
    return max(
        0.0,
        float((item or {}).get("real_energy", 0.0) or 0.0)
        + float((item or {}).get("virtual_energy", 0.0) or 0.0) * 0.35
        + float((item or {}).get("attention_gain", 0.0) or 0.0) * 0.5,
    )


def _int_value(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_value(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ShortTermEchoBuffer:
    """
    Bounded "afterimage / aftersound / recent-thought" buffer.

    AP needs recent inputs and thoughts to linger like human short-term
    sensory memory, but the echo must not pretend to be a new external input.
    This buffer therefore emits low-energy copies of already-seen SA plus
    explicit `echo::*` marker SA. Runtime can feed those into the normal state
    pool before recall, while trace/meta still shows that the source is echo.
    """

    def __init__(
        self,
        *,
        history_limit: int = 32,
        max_age_ticks: int = 8,
        decay: float = 0.68,
        sensory_gain: float = 0.22,
        thought_gain: float = 0.18,
        max_echo_energy: float = 0.28,
        max_items_per_tick: int = 18,
        modality_policies: dict[str, dict] | None = None,
    ) -> None:
        self._events: deque[dict] = deque(maxlen=max(1, int(history_limit)))
        self.max_age_ticks = max(1, int(max_age_ticks))
        self.decay = max(0.0, min(0.98, float(decay)))
        self.sensory_gain = max(0.0, float(sensory_gain))
        self.thought_gain = max(0.0, float(thought_gain))
        self.max_echo_energy = max(0.01, float(max_echo_energy))
        self.max_items_per_tick = max(1, int(max_items_per_tick))
        self.modality_policies = self._normalize_modality_policies(modality_policies or {})

    def observe_sensory_items(self, items: list[dict], *, tick_index: int) -> None:
        self._observe_items(items, tick_index=tick_index, echo_kind="sensory_echo")

    def observe_thought_items(self, items: list[dict], *, tick_index: int) -> None:
        self._observe_items(items, tick_index=tick_index, echo_kind="thought_echo")

    def build_echo_items(self, *, tick_index: int) -> dict:
        now_tick = int(tick_index)
        rows: list[tuple[float, dict, dict]] = []
        for event in list(self._events):
            origin_tick = _int_value(event.get("origin_tick_index"), now_tick)
            age = max(0, now_tick - origin_tick)
            policy = self._policy_for_event(event)
            if age <= 0 or age > int(policy["max_age_ticks"]):
                continue
            original = dict(event.get("item", {}) or {})
            if not original:
                continue
            echo_energy = min(
                float(policy["max_energy"]),
                _energy_of(original) * float(policy["gain"]) * (float(policy["decay"]) ** (age - 1)),
            )
            if echo_energy <= 0.0:
                continue
            echo_meta = self._echo_meta(event=event, age=age, echo_energy=echo_energy, policy=policy)
            rows.append((echo_energy, original, echo_meta))
        rows.sort(key=self._echo_sort_key)
        selected = self._select_echo_rows(rows)
        echo_items: list[dict] = []
        source_counter: Counter[str] = Counter()
        for echo_energy, original, echo_meta in selected:
            source_counter[str(echo_meta.get("echo_modality", "unknown") or "unknown")] += 1
            echo_items.append(self._echo_copy_item(original, echo_meta=echo_meta, echo_energy=echo_energy))
            echo_items.append(self._echo_marker_item(original, echo_meta=echo_meta, echo_energy=echo_energy))
        return {
            "schema_id": "short_term_echo_trace/v1",
            "tick_index": now_tick,
            "applied": bool(echo_items),
            "echo_count": len(echo_items),
            "source_counts": dict(source_counter),
            "items": echo_items,
            "items_preview": [
                {
                    "sa_label": str(item.get("sa_label", "") or ""),
                    "source_type": str(item.get("source_type", "") or ""),
                    "family": str(item.get("family", "") or ""),
                    "real_energy": _round4(float(item.get("real_energy", 0.0) or 0.0)),
                    "echo_modality": str((item.get("anchor_meta", {}) or {}).get("echo_modality", "") or ""),
                    "age_ticks": int((item.get("anchor_meta", {}) or {}).get("age_ticks", 0) or 0),
                    "not_new_external_input": bool((item.get("anchor_meta", {}) or {}).get("not_new_external_input", False)),
                }
                for item in echo_items[: min(10, len(echo_items))]
            ],
            "policy": "decayed_short_term_echo_not_new_external_input",
            "lifespan_policy": self._trace_lifespan_policy(),
        }

    def _normalize_modality_policies(self, policies: dict[str, dict]) -> dict[str, dict]:
        defaults = {
            "vision": {
                "max_age_ticks": min(self.max_age_ticks, 4),
                "decay": min(self.decay, 0.42),
                "sensory_gain": min(self.sensory_gain, 0.18),
                "thought_gain": min(self.thought_gain, 0.15),
                "max_energy": min(self.max_echo_energy, 0.16),
            },
            "audio": {
                "max_age_ticks": max(self.max_age_ticks, 24),
                "decay": max(self.decay, 0.82),
                "sensory_gain": min(self.sensory_gain, 0.20),
                "thought_gain": min(self.thought_gain, 0.15),
                "max_energy": min(self.max_echo_energy, 0.22),
            },
            "text": {
                "max_age_ticks": max(self.max_age_ticks, 10),
                "decay": max(self.decay, 0.72),
                "sensory_gain": min(self.sensory_gain, 0.18),
                "thought_gain": min(self.thought_gain, 0.15),
                "max_energy": min(self.max_echo_energy, 0.18),
            },
            "thought": {
                "max_age_ticks": max(self.max_age_ticks, 14),
                "decay": max(self.decay, 0.76),
                "sensory_gain": min(self.sensory_gain, 0.15),
                "thought_gain": min(self.thought_gain, 0.15),
                "max_energy": min(self.max_echo_energy, 0.16),
            },
        }
        normalized: dict[str, dict] = {}
        for modality, default in defaults.items():
            override = dict((policies or {}).get(modality, {}) or {})
            normalized[modality] = {
                "max_age_ticks": max(1, _int_value(override.get("max_age_ticks"), int(default["max_age_ticks"]))),
                "decay": max(0.0, min(0.98, _float_value(override.get("decay"), float(default["decay"])))),
                "sensory_gain": max(0.0, _float_value(override.get("sensory_gain", override.get("gain")), float(default["sensory_gain"]))),
                "thought_gain": max(0.0, _float_value(override.get("thought_gain", override.get("gain")), float(default["thought_gain"]))),
                "max_energy": max(0.01, _float_value(override.get("max_energy"), float(default["max_energy"]))),
            }
        return normalized

    def _policy_for_event(self, event: dict) -> dict:
        modality = str((event or {}).get("modality", "thought") or "thought")
        echo_kind = str((event or {}).get("echo_kind", "sensory_echo") or "sensory_echo")
        base = dict(self.modality_policies.get(modality) or self.modality_policies.get("thought") or {})
        gain_key = "thought_gain" if echo_kind == "thought_echo" else "sensory_gain"
        gain = float(base.get(gain_key, self.thought_gain if echo_kind == "thought_echo" else self.sensory_gain) or 0.0)
        return {
            "modality": modality,
            "echo_kind": echo_kind,
            "max_age_ticks": int(base.get("max_age_ticks", self.max_age_ticks) or self.max_age_ticks),
            "decay": float(base.get("decay", self.decay) or self.decay),
            "gain": gain,
            "max_energy": float(base.get("max_energy", self.max_echo_energy) or self.max_echo_energy),
            "policy": "humanlike_modality_specific_echo_lifespan",
        }

    def _trace_lifespan_policy(self) -> dict:
        return {
            modality: {
                "max_age_ticks": int(policy.get("max_age_ticks", 0) or 0),
                "decay": _round4(float(policy.get("decay", 0.0) or 0.0)),
                "sensory_gain": _round4(float(policy.get("sensory_gain", 0.0) or 0.0)),
                "thought_gain": _round4(float(policy.get("thought_gain", 0.0) or 0.0)),
                "max_energy": _round4(float(policy.get("max_energy", 0.0) or 0.0)),
            }
            for modality, policy in sorted(self.modality_policies.items())
        }

    def _select_echo_rows(self, rows: list[tuple[float, dict, dict]]) -> list[tuple[float, dict, dict]]:
        """
        Keep short-term echo multimodal instead of letting one loud channel win.

        A human afterimage/aftersound/recent-thought field often has several
        weak residues at once. We therefore reserve one representative for each
        (echo_kind, modality) group that is already present, then fill the rest
        by energy. This is a channel-level budget policy, not a content rule.
        """

        limit = max(1, int(self.max_items_per_tick))
        selected: list[tuple[float, dict, dict]] = []
        selected_keys: set[tuple[str, str, str, int]] = set()
        grouped: dict[tuple[str, str], list[tuple[float, dict, dict]]] = {}
        for row in rows:
            meta = dict(row[2] or {})
            group_key = (
                str(meta.get("echo_kind", "") or ""),
                str(meta.get("echo_modality", "") or ""),
            )
            grouped.setdefault(group_key, []).append(row)
        for group_key in sorted(grouped):
            group_rows = sorted(grouped[group_key], key=self._echo_sort_key)
            if not group_rows:
                continue
            row = group_rows[0]
            key = self._row_identity(row)
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)
            if len(selected) >= limit:
                return selected[:limit]
        for row in rows:
            key = self._row_identity(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _echo_sort_key(self, row: tuple[float, dict, dict]) -> tuple:
        energy, original, meta = row
        # Prefer rows that can reconstruct perceptual residue when energies tie.
        has_payload = 1 if isinstance((original or {}).get("reconstruction_payload"), dict) and (original or {}).get("reconstruction_payload") else 0
        return (
            -float(energy),
            -has_payload,
            str((meta or {}).get("echo_kind", "") or ""),
            str((meta or {}).get("echo_modality", "") or ""),
            str((original or {}).get("sa_label", "") or ""),
        )

    def _row_identity(self, row: tuple[float, dict, dict]) -> tuple[str, str, str, int]:
        _energy, original, meta = row
        return (
            str((meta or {}).get("echo_kind", "") or ""),
            str((meta or {}).get("echo_modality", "") or ""),
            str((original or {}).get("sa_label", "") or ""),
            _int_value((meta or {}).get("origin_tick_index"), -1),
        )

    def _observe_items(self, items: list[dict], *, tick_index: int, echo_kind: str) -> None:
        seen: set[str] = set()
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            if self._is_echo_item(item):
                continue
            modality = self._modality_for_item(item)
            if modality == "other":
                continue
            energy = _energy_of(item)
            if energy <= 0.0 and str(echo_kind) != "thought_echo":
                continue
            stored_modality = self._stored_modality_for_echo(item, echo_kind=echo_kind, modality=modality)
            keep_reconstruction_payload = self._should_keep_reconstruction_payload(echo_kind=echo_kind, modality=modality)
            seen.add(label)
            self._events.append(
                {
                    "schema_id": "short_term_echo_event/v1",
                    "origin_tick_index": int(tick_index),
                    "echo_kind": str(echo_kind),
                    "modality": stored_modality,
                    "original_modality": modality,
                    "original_source_type": str(item.get("source_type", "") or ""),
                    "item": self._compact_item(item, keep_reconstruction_payload=keep_reconstruction_payload),
                }
            )

    def _stored_modality_for_echo(self, item: dict, *, echo_kind: str, modality: str) -> str:
        """
        Keep sensory residue separate from "I was just thinking about X".

        If attention keeps selecting a visual/audio object during blank ticks,
        that is an internal thought trace, not a renewed retinal/aural residue.
        We store such entries as thought echo and keep the original modality in
        metadata, preventing perceptual payloads from being refreshed forever.
        """

        if str(echo_kind) == "thought_echo" and str(modality) in {"vision", "audio"}:
            return "thought"
        return str(modality or "thought")

    def _should_keep_reconstruction_payload(self, *, echo_kind: str, modality: str) -> bool:
        # Low-level visual/audio payloads belong to sensory echo. A thought about
        # a visual/audio object may still preserve the SA label, but not the full
        # perceptual reconstruction payload that would render as afterimage or
        # aftersound drag in the observatory.
        if str(echo_kind) == "thought_echo" and str(modality) in {"vision", "audio"}:
            return False
        return True

    def _compact_item(self, item: dict, *, keep_reconstruction_payload: bool = True) -> dict:
        row = {
            "sa_label": str(item.get("sa_label", "") or ""),
            "display_text": str(item.get("display_text", item.get("sa_label", "")) or item.get("sa_label", "")),
            "family": str(item.get("family", "echo") or "echo"),
            "source_type": str(item.get("source_type", "") or ""),
            "real_energy": max(0.0, float(item.get("real_energy", 0.0) or 0.0)),
            "virtual_energy": max(0.0, float(item.get("virtual_energy", 0.0) or 0.0)),
            "attention_gain": max(0.0, float(item.get("attention_gain", item.get("focus_score", item.get("attention_score", 0.0))) or 0.0)),
            "position": item.get("position", 0),
        }
        if isinstance(item.get("anchor_meta"), dict):
            row["anchor_meta"] = dict(item.get("anchor_meta", {}) or {})
        if isinstance(item.get("numeric_features"), dict):
            row["numeric_features"] = {
                str(channel): list(values if isinstance(values, (list, tuple)) else [values])
                for channel, values in dict(item.get("numeric_features", {}) or {}).items()
                if str(channel or "")
            }
        if keep_reconstruction_payload and isinstance(item.get("reconstruction_payload"), dict):
            row["reconstruction_payload"] = dict(item.get("reconstruction_payload", {}) or {})
        return row

    def _echo_meta(self, *, event: dict, age: int, echo_energy: float, policy: dict | None = None) -> dict:
        origin_tick = _int_value(event.get("origin_tick_index"), 0)
        modality = str(event.get("modality", "unknown") or "unknown")
        original_modality = str(event.get("original_modality", modality) or modality)
        echo_kind = str(event.get("echo_kind", "sensory_echo") or "sensory_echo")
        clean_policy = dict(policy or {})
        return {
            "schema_id": "short_term_echo_meta/v1",
            "is_echo": True,
            "echo_kind": echo_kind,
            "echo_modality": modality,
            "original_modality": original_modality,
            "origin_tick_index": origin_tick,
            "age_ticks": int(age),
            "echo_energy": _round4(echo_energy),
            "policy_max_age_ticks": int(clean_policy.get("max_age_ticks", self.max_age_ticks) or self.max_age_ticks),
            "policy_decay": _round4(float(clean_policy.get("decay", self.decay) or self.decay)),
            "policy_gain": _round4(float(clean_policy.get("gain", 0.0) or 0.0)),
            "policy_max_energy": _round4(float(clean_policy.get("max_energy", self.max_echo_energy) or self.max_echo_energy)),
            "modality_lifespan_policy": str(clean_policy.get("policy", "humanlike_modality_specific_echo_lifespan") or "humanlike_modality_specific_echo_lifespan"),
            "original_source_type": str(event.get("original_source_type", "") or ""),
            "not_new_external_input": True,
            "meaning": "short_term_afterimage_or_recent_thought_not_current_external_input",
        }

    def _echo_copy_item(self, original: dict, *, echo_meta: dict, echo_energy: float) -> dict:
        row = dict(original)
        original_label = str(row.get("sa_label", "") or "")
        row["display_text"] = str(row.get("display_text", original_label) or original_label)
        row["source_type"] = str(echo_meta.get("echo_kind", "sensory_echo") or "sensory_echo")
        row["real_energy"] = _round4(echo_energy)
        row["virtual_energy"] = 0.0
        row["attention_gain"] = _round4(echo_energy * 0.55)
        row["cognitive_pressure"] = _round4(echo_energy)
        meta = dict(row.get("anchor_meta", {}) or {})
        meta.update(echo_meta)
        meta["echo_target_label"] = original_label
        row["anchor_meta"] = meta
        return row

    def _echo_marker_item(self, original: dict, *, echo_meta: dict, echo_energy: float) -> dict:
        original_label = str(original.get("sa_label", "") or "unknown")
        modality = str(echo_meta.get("echo_modality", "unknown") or "unknown")
        kind = str(echo_meta.get("echo_kind", "sensory_echo") or "sensory_echo")
        marker = {
            "sa_label": f"echo::{kind}::{modality}::{original_label}",
            "display_text": f"echo {modality} {original.get('display_text', original_label)}",
            "family": "short_term_echo",
            "source_type": kind,
            "real_energy": _round4(echo_energy * 0.62),
            "virtual_energy": 0.0,
            "attention_gain": _round4(echo_energy * 0.35),
            "anchor_meta": dict(echo_meta) | {"echo_target_label": original_label},
        }
        if isinstance(original.get("numeric_features"), dict):
            marker["numeric_features"] = dict(original.get("numeric_features", {}) or {})
        return marker

    def _is_echo_item(self, item: dict) -> bool:
        source = str((item or {}).get("source_type", "") or "")
        if source in {"sensory_echo", "thought_echo"}:
            return True
        label = str((item or {}).get("sa_label", "") or "")
        if label.startswith("echo::"):
            return True
        return bool(((item or {}).get("anchor_meta", {}) or {}).get("is_echo", False))

    def _modality_for_item(self, item: dict) -> str:
        source = str((item or {}).get("source_type", "") or "")
        family = str((item or {}).get("family", "") or "")
        label = str((item or {}).get("sa_label", "") or "")
        if source == "external_text" or family in {"text", "learned_text_phrase"} or label.startswith(("text::", "phrase::")):
            return "text"
        if source.startswith("vision") or family.startswith("vision") or label.startswith(("vision::", "vision_obj::")):
            return "vision"
        if source.startswith("audio") or family.startswith("audio") or label.startswith(("audio::", "audio_event::")):
            return "audio"
        if source in {"focus_continuation", "focus_replay", "action_control"}:
            return "thought"
        if family in {"cognitive_feeling", "expectation_pressure", "time_feeling", "rhythm_feeling"}:
            return "thought"
        return "thought" if item.get("is_focus") else "other"
