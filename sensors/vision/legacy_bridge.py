from __future__ import annotations

from legacy_apv2.sensors.vision_sensor_v1 import VisionSensorV1


def _round4(value: float) -> float:
    return round(float(value), 4)


class LegacyVisionBridge:
    def __init__(self) -> None:
        self.sensor = VisionSensorV1(patch_budget=24, focus_patch_budget=12)

    def ingest_image_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "image_input") -> dict:
        packet = self.sensor.ingest_image_bytes(raw_bytes, tick_index=tick_index, source_type=source_type)
        state_items = []
        for item in (packet.get("focus_priority_samples", []) or [])[:24]:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            attrs = dict(item.get("attributes", {}) or {})
            coords = dict(item.get("coords", {}) or {})
            energy = float(item.get("energy", 0.0) or 0.0)
            focus_priority = float(attrs.get("focus_priority", 0.0) or 0.0)
            state_items.append(
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "source_type": "vision_bridge",
                    "family": "vision",
                    "real_energy": _round4(max(0.05, energy * (0.65 + 0.35 * focus_priority))),
                    "anchor_meta": {
                        "channel": "vision",
                        "legacy_sensor": "vision_sensor_v1",
                        "coords": coords,
                        "attributes": {
                            "focus_priority": _round4(focus_priority),
                            "edge_strength": _round4(float(attrs.get("edge_strength", 0.0) or 0.0)),
                            "motion": _round4(float(attrs.get("motion", 0.0) or 0.0)),
                            "memory_feature_code": str(attrs.get("memory_feature_code", "") or ""),
                            "sample_reason": str(attrs.get("sample_reason", "") or ""),
                        },
                    },
                }
            )
        for item in (packet.get("dynamic_motion_samples", []) or [])[:8]:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            attrs = dict(item.get("attributes", {}) or {})
            state_items.append(
                {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", label) or label),
                    "source_type": "vision_bridge_dynamic",
                    "family": "vision_dynamic",
                    "real_energy": _round4(max(0.08, float(item.get("energy", 0.0) or 0.0))),
                    "anchor_meta": {
                        "channel": "vision",
                        "legacy_sensor": "vision_sensor_v1",
                        "dynamic_objectness": _round4(float(attrs.get("dynamic_objectness", 0.0) or 0.0)),
                        "temporal_persistence": _round4(float(attrs.get("temporal_persistence", 0.0) or 0.0)),
                        "motion_speed": _round4(float(attrs.get("motion_speed", 0.0) or 0.0)),
                        "family_key": str(attrs.get("track_id", label) or label),
                    },
                }
            )
        inner_view = {
            "current_frame": dict(packet.get("preview_image", {}) or {}),
            "recall_layers": [],
            "prediction_layers": [],
            "focus_objects": [str(item.get("sa_label", "") or "") for item in (packet.get("focus_priority_samples", []) or [])[:8] if str(item.get("sa_label", "") or "")],
            "energy_summary": {
                "budget_used": int(packet.get("budget_used", 0) or 0),
                "focus_priority_budget": int(packet.get("focus_priority_budget", 0) or 0),
                "dynamic_track_count": int(((packet.get("dynamic_track_summary", {}) or {}).get("track_count", 0) or 0)),
                "dynamic_object_count": int(((packet.get("dynamic_track_summary", {}) or {}).get("object_count", 0) or 0)),
            },
            "contour_reconstruction": dict(packet.get("contour_reconstruction", {}) or {}),
        }
        return {
            "packet": packet,
            "state_items": state_items,
            "inner_vision": inner_view,
        }
