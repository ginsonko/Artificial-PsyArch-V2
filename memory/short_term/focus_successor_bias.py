from __future__ import annotations

import math
from collections import OrderedDict


def _round4(value: float) -> float:
    return round(float(value), 4)


class FocusSuccessorBias:
    """
    Online, bounded successor bias for the slow-system focus stream.

    This is not a replacement for Bn'/Cn'. It learns a small statistical bias
    from the previous focus window to the next real-energy focus objects so the
    next empty tick can feel like continuation rather than mere persistence.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        context_limit: int = 2048,
        max_successors_per_context: int = 64,
        max_context_labels: int = 8,
        max_order: int = 3,
        top_k: int = 12,
        per_tick_update_limit: int = 16,
        real_threshold: float = 0.08,
        decay: float = 0.992,
        rescale_threshold: float = 64.0,
        rescale_factor: float = 0.5,
        min_support: float = 0.18,
        gain: float = 0.42,
        max_bias: float = 0.48,
        entropy_floor: float = 0.28,
    ) -> None:
        self.enabled = bool(enabled)
        self.context_limit = max(16, int(context_limit))
        self.max_successors_per_context = max(4, int(max_successors_per_context))
        self.max_context_labels = max(1, int(max_context_labels))
        self.max_order = max(1, int(max_order))
        self.top_k = max(1, int(top_k))
        self.per_tick_update_limit = max(1, int(per_tick_update_limit))
        self.real_threshold = max(0.0, float(real_threshold))
        self.decay = max(0.0, min(1.0, float(decay)))
        self.rescale_threshold = max(1.0, float(rescale_threshold))
        self.rescale_factor = max(0.05, min(1.0, float(rescale_factor)))
        self.min_support = max(0.01, float(min_support))
        self.gain = max(0.0, float(gain))
        self.max_bias = max(0.0, float(max_bias))
        self.entropy_floor = max(0.0, min(1.0, float(entropy_floor)))
        self._contexts: OrderedDict[str, dict] = OrderedDict()
        self._current_tick = -1
        self._updates_this_tick = 0
        self._last_learning_events: list[dict] = []
        self._last_bias_trace: dict = self._empty_bias_trace(tick_index=-1)

    def begin_tick(self, tick_index: int) -> None:
        if int(tick_index) != self._current_tick:
            self._current_tick = int(tick_index)
            self._updates_this_tick = 0

    def build_bias(self, *, previous_focus_labels: list[str], candidate_items: list[dict], tick_index: int) -> dict:
        self.begin_tick(tick_index)
        if not self.enabled:
            trace = self._empty_bias_trace(tick_index=tick_index)
            trace["disabled"] = True
            self._last_bias_trace = trace
            return trace
        candidate_labels = self._candidate_labels(candidate_items)
        context_specs = self._context_specs(previous_focus_labels)
        if not candidate_labels or not context_specs:
            trace = self._empty_bias_trace(tick_index=tick_index)
            trace["source_focus_labels"] = self._clean_labels(previous_focus_labels)[: self.max_context_labels]
            self._last_bias_trace = trace
            return trace

        by_label: dict[str, dict] = {}
        context_traces = []
        for spec in context_specs:
            context = self._contexts.get(spec["key"])
            if not context:
                continue
            self._touch_context(spec["key"], tick_index=tick_index)
            context = self._contexts.get(spec["key"])
            if not context:
                continue
            total = max(1e-9, float(context.get("total", 0.0) or 0.0))
            successors = dict(context.get("successors", {}) or {})
            entropy = self._entropy(successors, total=total)
            branch_damping = max(self.entropy_floor, 1.0 - 0.72 * entropy)
            matched = 0
            for label in candidate_labels:
                row = successors.get(label)
                if not row:
                    continue
                count = float(row.get("weight", 0.0) or 0.0)
                if count <= 0.0:
                    continue
                normalized = count / total
                support_gate = min(1.0, count / self.min_support)
                contribution = normalized * support_gate * branch_damping * float(spec.get("weight", 1.0) or 1.0)
                if contribution <= 0.0:
                    continue
                bucket = by_label.setdefault(
                    label,
                    {
                        "sa_label": label,
                        "raw": 0.0,
                        "support": 0.0,
                        "contexts": [],
                    },
                )
                bucket["raw"] = float(bucket["raw"]) + contribution
                bucket["support"] = max(float(bucket["support"]), count)
                bucket["contexts"].append(
                    {
                        "context_key": spec["key"],
                        "context_labels": list(spec["labels"]),
                        "count": _round4(count),
                        "normalized": _round4(normalized),
                        "branch_entropy": _round4(entropy),
                        "branch_damping": _round4(branch_damping),
                        "contribution": _round4(contribution),
                    }
                )
                matched += 1
            if matched:
                context_traces.append(
                    {
                        "context_key": spec["key"],
                        "context_labels": list(spec["labels"]),
                        "total": _round4(total),
                        "branch_entropy": _round4(entropy),
                        "branch_damping": _round4(branch_damping),
                        "matched_candidates": matched,
                    }
                )

        rows = []
        bias_by_label = {}
        for label, row in by_label.items():
            raw = float(row.get("raw", 0.0) or 0.0)
            bias = min(self.max_bias, raw * self.gain)
            if bias <= 0.0:
                continue
            rows.append(
                {
                    "sa_label": label,
                    "bias": _round4(bias),
                    "raw": _round4(raw),
                    "support": _round4(float(row.get("support", 0.0) or 0.0)),
                    "contexts": list(row.get("contexts", []) or [])[:4],
                }
            )
            bias_by_label[label] = _round4(bias)
        rows.sort(key=lambda item: (-float(item.get("bias", 0.0) or 0.0), str(item.get("sa_label", "") or "")))
        rows = rows[: self.top_k]
        bias_by_label = {str(row["sa_label"]): float(row["bias"]) for row in rows}
        trace = {
            "schema_id": "focus_successor_bias/v1",
            "tick_index": int(tick_index),
            "source_focus_labels": self._clean_labels(previous_focus_labels)[: self.max_context_labels],
            "candidate_count": len(candidate_labels),
            "context_count": len(context_specs),
            "matched_context_count": len(context_traces),
            "items": rows,
            "bias_by_label": bias_by_label,
            "context_traces": context_traces[:6],
            "learning": self.summary(),
        }
        self._last_bias_trace = trace
        return trace

    def observe_transition(self, *, previous_focus_labels: list[str], current_focus_items: list[dict], tick_index: int) -> dict:
        self.begin_tick(tick_index)
        self._last_learning_events = []
        if not self.enabled:
            return self._learning_trace(tick_index=tick_index, source_focus_labels=previous_focus_labels)
        context_specs = self._context_specs(previous_focus_labels)
        targets = self._target_rows(current_focus_items)
        if not context_specs or not targets:
            return self._learning_trace(tick_index=tick_index, source_focus_labels=previous_focus_labels)
        previous_set = set(self._clean_labels(previous_focus_labels))
        for spec in context_specs:
            if self._updates_this_tick >= self.per_tick_update_limit:
                break
            context = self._touch_context(spec["key"], tick_index=tick_index, labels=spec["labels"])
            updated = 0
            for target in targets:
                if self._updates_this_tick >= self.per_tick_update_limit:
                    break
                label = str(target.get("sa_label", "") or "")
                if not label or label in previous_set:
                    continue
                weight = self._event_amount(float(target.get("real_energy", 0.0) or 0.0)) * float(spec.get("weight", 1.0) or 1.0)
                if weight <= 0.0:
                    continue
                successors = context.setdefault("successors", OrderedDict())
                if label not in successors and len(successors) >= self.max_successors_per_context:
                    self._evict_weakest_successor(successors)
                row = successors.setdefault(label, {"weight": 0.0, "last_tick": int(tick_index)})
                row["weight"] = float(row.get("weight", 0.0) or 0.0) + weight
                row["last_tick"] = int(tick_index)
                context["total"] = float(context.get("total", 0.0) or 0.0) + weight
                updated += 1
                self._updates_this_tick += 1
                self._last_learning_events.append(
                    {
                        "context_key": spec["key"],
                        "context_labels": list(spec["labels"]),
                        "successor_label": label,
                        "weight": _round4(weight),
                        "target_real_energy": _round4(float(target.get("real_energy", 0.0) or 0.0)),
                    }
                )
            if updated:
                self._rescale_if_needed(context)
        return self._learning_trace(tick_index=tick_index, source_focus_labels=previous_focus_labels)

    def summary(self) -> dict:
        successor_count = 0
        total_support = 0.0
        for context in self._contexts.values():
            successors = dict(context.get("successors", {}) or {})
            successor_count += len(successors)
            total_support += float(context.get("total", 0.0) or 0.0)
        return {
            "schema_id": "focus_successor_bias_learning/v1",
            "enabled": self.enabled,
            "context_count": len(self._contexts),
            "successor_edge_count": successor_count,
            "total_support": _round4(total_support),
            "context_limit": self.context_limit,
            "max_successors_per_context": self.max_successors_per_context,
            "per_tick_update_limit": self.per_tick_update_limit,
            "last_learning_events": [dict(row) for row in self._last_learning_events[:8]],
        }

    def _empty_bias_trace(self, *, tick_index: int) -> dict:
        return {
            "schema_id": "focus_successor_bias/v1",
            "tick_index": int(tick_index),
            "source_focus_labels": [],
            "candidate_count": 0,
            "context_count": 0,
            "matched_context_count": 0,
            "items": [],
            "bias_by_label": {},
            "context_traces": [],
            "learning": self.summary() if hasattr(self, "_contexts") else {},
        }

    def _learning_trace(self, *, tick_index: int, source_focus_labels: list[str]) -> dict:
        return {
            "schema_id": "focus_successor_bias_update/v1",
            "tick_index": int(tick_index),
            "source_focus_labels": self._clean_labels(source_focus_labels)[: self.max_context_labels],
            "updated_event_count": len(self._last_learning_events),
            "events": [dict(row) for row in self._last_learning_events[:8]],
            "summary": self.summary(),
        }

    def _touch_context(self, key: str, *, tick_index: int, labels: list[str] | None = None) -> dict:
        clean = str(key or "")
        if not clean:
            clean = "empty"
        if clean not in self._contexts and len(self._contexts) >= self.context_limit:
            self._contexts.popitem(last=False)
        context = self._contexts.setdefault(
            clean,
            {
                "context_key": clean,
                "labels": list(labels or []),
                "total": 0.0,
                "successors": OrderedDict(),
                "last_tick": int(tick_index),
            },
        )
        self._contexts.move_to_end(clean)
        if labels:
            context["labels"] = list(labels)
        last_tick = int(context.get("last_tick", tick_index) if context.get("last_tick", None) is not None else tick_index)
        age = max(0, int(tick_index) - last_tick)
        if age > 0 and self.decay < 1.0:
            scale = self.decay ** age
            context["total"] = float(context.get("total", 0.0) or 0.0) * scale
            successors = context.setdefault("successors", OrderedDict())
            for label in list(successors.keys()):
                row = successors[label]
                row["weight"] = float(row.get("weight", 0.0) or 0.0) * scale
                if float(row.get("weight", 0.0) or 0.0) < self.min_support * 0.05:
                    successors.pop(label, None)
            context["total"] = sum(float(row.get("weight", 0.0) or 0.0) for row in successors.values())
        context["last_tick"] = int(tick_index)
        return context

    def _context_specs(self, labels: list[str]) -> list[dict]:
        clean = self._clean_labels(labels)[-self.max_context_labels :]
        if not clean:
            return []
        specs = []
        for idx, label in enumerate(clean):
            distance_from_tail = len(clean) - 1 - idx
            specs.append(
                {
                    "key": f"label::{label}",
                    "labels": [label],
                    "weight": 0.72 ** distance_from_tail,
                }
            )
        max_order = min(self.max_order, len(clean))
        for order in range(2, max_order + 1):
            tail = clean[-order:]
            specs.append(
                {
                    "key": "tail::" + ">>".join(tail),
                    "labels": tail,
                    "weight": 0.86 + 0.12 * order,
                }
            )
        return specs

    def _candidate_labels(self, candidate_items: list[dict]) -> list[str]:
        labels = []
        seen = set()
        for item in candidate_items or []:
            label = str((item or {}).get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    def _target_rows(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or self._is_non_core_target(label, item):
                continue
            real = float(item.get("real_energy", 0.0) or 0.0)
            if real < self.real_threshold:
                continue
            rows.append({"sa_label": label, "real_energy": real})
            if len(rows) >= self.top_k:
                break
        return rows

    def _clean_labels(self, labels: list[str]) -> list[str]:
        rows = []
        seen = set()
        for label in labels or []:
            clean = str(label or "")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
        return rows

    def _is_non_core_target(self, label: str, item: dict) -> bool:
        clean = str(label or "")
        if clean.startswith(("feeling::", "timefelt::", "rhythmfelt::", "expectation_pressure::")):
            return True
        if clean.startswith(("action::", "action_feedback::", "text_action::")):
            return True
        family = str((item or {}).get("family", "") or "")
        if family in {"cognitive_feeling", "action", "action_feedback"}:
            return True
        return False

    def _event_amount(self, weight: float) -> float:
        value = max(0.0, float(weight or 0.0))
        if value <= 0.0:
            return 0.0
        return max(0.03, min(1.6, value))

    def _rescale_if_needed(self, context: dict) -> None:
        total = float(context.get("total", 0.0) or 0.0)
        if total <= self.rescale_threshold:
            return
        successors = context.setdefault("successors", OrderedDict())
        for row in successors.values():
            row["weight"] = float(row.get("weight", 0.0) or 0.0) * self.rescale_factor
        context["total"] = sum(float(row.get("weight", 0.0) or 0.0) for row in successors.values())

    def _evict_weakest_successor(self, successors: OrderedDict[str, dict]) -> None:
        weakest = None
        weakest_weight = None
        for label, row in successors.items():
            weight = float(row.get("weight", 0.0) or 0.0)
            if weakest is None or weight < float(weakest_weight or 0.0):
                weakest = label
                weakest_weight = weight
        if weakest is not None:
            successors.pop(weakest, None)

    def _entropy(self, successors: dict, *, total: float) -> float:
        weights = [float(row.get("weight", 0.0) or 0.0) for row in successors.values() if float(row.get("weight", 0.0) or 0.0) > 0.0]
        if len(weights) <= 1 or total <= 1e-9:
            return 0.0
        entropy = 0.0
        for weight in weights:
            p = weight / total
            if p > 0.0:
                entropy -= p * math.log(p)
        max_entropy = math.log(len(weights))
        if max_entropy <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, entropy / max_entropy))
