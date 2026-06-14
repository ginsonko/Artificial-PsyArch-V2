from __future__ import annotations

from legacy_apv2.sensors.hearing_sensor_v1 import HearingSensorV1


def _round4(value: float) -> float:
    return round(float(value), 4)


class LegacyAudioBridge:
    def __init__(self) -> None:
        self.sensor = HearingSensorV1(window_budget=12)

    def ingest_wav_bytes(self, raw_bytes: bytes, *, tick_index: int, source_type: str = "audio_input") -> dict:
        packet = self.sensor.ingest_wav_bytes(raw_bytes, tick_index=tick_index, source_type=source_type)
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
                    "source_type": "audio_bridge",
                    "family": "audio",
                    "real_energy": _round4(max(0.05, energy * (0.65 + 0.35 * focus_priority))),
                    "anchor_meta": {
                        "channel": "audio",
                        "legacy_sensor": "hearing_sensor_v1",
                        "coords": coords,
                        "attributes": {
                            "focus_priority": _round4(focus_priority),
                            "dominant_hz": _round4(float(attrs.get("dominant_hz", 0.0) or 0.0)),
                            "onset_strength": _round4(float(attrs.get("onset_strength", 0.0) or 0.0)),
                            "novelty": _round4(float(attrs.get("novelty", 0.0) or 0.0)),
                            "memory_feature_code": str(attrs.get("memory_feature_code", "") or ""),
                            "sample_reason": str(attrs.get("sample_reason", "") or ""),
                        },
                    },
                }
            )
        inner_view = {
            "current_bands": [str(item.get("sa_label", "") or "") for item in (packet.get("windows", []) or [])[:8] if str(item.get("sa_label", "") or "")],
            "primary_peaks": [str(item.get("sa_label", "") or "") for item in (packet.get("focus_priority_samples", []) or [])[:8] if str(item.get("sa_label", "") or "")],
            "recall_stream": [],
            "prediction_stream": [],
            "preview_asset_ref": {
                "preview_wav_b64": str(packet.get("preview_wav_b64", "") or ""),
                "proxy_preview_wav_b64": str(packet.get("proxy_preview_wav_b64", "") or ""),
                "preview_duration_ms": float(packet.get("preview_duration_ms", 0.0) or 0.0),
            },
            "energy_summary": {
                "budget_used": int(packet.get("budget_used", 0) or 0),
                "focus_band_budget": int(packet.get("focus_band_budget", 0) or 0),
                "dominant_hz": _round4(float(((packet.get("feature_summary", {}) or {}).get("dominant_hz", 0.0) or 0.0))),
                "tonal_clarity": _round4(float(((packet.get("feature_summary", {}) or {}).get("tonal_clarity", 0.0) or 0.0))),
            },
        }
        return {
            "packet": packet,
            "state_items": state_items,
            "inner_audio": inner_view,
        }
